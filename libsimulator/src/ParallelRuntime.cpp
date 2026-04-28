#include "ParallelRuntime.hpp"

#include <algorithm>

ParallelRuntime ParallelRuntime::instance{};

ParallelRuntime::ParallelRuntime()
{
	const char* env_threads = std::getenv("JPS_NUM_THREADS");
	if(env_threads != nullptr) {
		try {
			const auto parsed = std::stoul(std::string(env_threads));
			_num_threads = std::max<size_t>(1, parsed);
		} catch(...) {
			_num_threads = 1;
		}
	}

	startWorkers();
}

ParallelRuntime::~ParallelRuntime()
{
	stopWorkers();
}

void ParallelRuntime::setNumThreads(size_t num_threads)
{
	const size_t normalized = std::max<size_t>(1, num_threads);
	if(normalized == _num_threads) {
		return;
	}

	stopWorkers();
	_num_threads = normalized;
	startWorkers();
}

bool ParallelRuntime::enqueue(std::function<void()> task)
{
	if(!task) {
		return true;
	}

	{
		std::lock_guard<std::mutex> lock(_queue_mutex);
		if(_stop_requested) {
			// Runtime is stopping; signal failure to caller so it may run
			// the work inline instead of throwing while local synchronization
			// objects are in scope.
			return false;
		}
		_task_queue.push(std::move(task));
	}
	_queue_cv.notify_one();
	return true;
}

void ParallelRuntime::startWorkers()
{
	{
		std::lock_guard<std::mutex> lock(_queue_mutex);
		_stop_requested = false;
	}

	_workers.clear();
	_workers.reserve(_num_threads);
	for(size_t i = 0; i < _num_threads; ++i) {
		_workers.emplace_back([this]() { workerLoop(); });
	}
}

void ParallelRuntime::stopWorkers()
{
	{
		std::lock_guard<std::mutex> lock(_queue_mutex);
		_stop_requested = true;
	}
	_queue_cv.notify_all();

	for(auto& worker : _workers) {
		if(worker.joinable()) {
			worker.join();
		}
	}
	_workers.clear();

	std::queue<std::function<void()>> empty;
	{
		std::lock_guard<std::mutex> lock(_queue_mutex);
		_task_queue.swap(empty);
	}
}

void ParallelRuntime::workerLoop()
{
	while(true) {
		std::function<void()> task;
		{
			std::unique_lock<std::mutex> lock(_queue_mutex);
			_queue_cv.wait(lock, [this]() { return _stop_requested || !_task_queue.empty(); });

			if(_stop_requested && _task_queue.empty()) {
				return;
			}

			task = std::move(_task_queue.front());
			_task_queue.pop();
		}

		task();
	}
}
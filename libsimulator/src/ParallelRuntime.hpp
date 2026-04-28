// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <cstdlib>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <execution>

class ParallelRuntime {

    static ParallelRuntime instance;
    size_t _num_threads{1};
    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _task_queue;
    mutable std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    bool _stop_requested{false};

public:
    ParallelRuntime();
    ~ParallelRuntime();

    static ParallelRuntime& Instance()
    {
        return instance;
    }

    ParallelRuntime(const ParallelRuntime& other) = delete;
    ParallelRuntime& operator=(const ParallelRuntime& other) = delete;
    ParallelRuntime(ParallelRuntime&& other) = delete;
    ParallelRuntime& operator=(ParallelRuntime&& other) = delete;

    void setNumThreads(size_t num_threads);

    size_t getNumThreads() const
    {
        return _num_threads;
    }

    // Try to enqueue a task. Returns true on success. If the runtime is
    // stopping this will return false instead of throwing so callers can
    // safely fall back to running the work inline and avoid unwinding while
    // local synchronization objects are alive.
    bool enqueue(std::function<void()> task);

    template <typename F>
    void parallelFor(size_t begin, size_t end, F&& fn)
    {
        if(end <= begin) {
            return;
        }

        const size_t work_items = end - begin;
        if(_num_threads <= 1 || work_items <= 1) {
            for(size_t i = begin; i < end; ++i) {
                fn(i);
            }
            return;
        }

        const size_t task_count = std::min(work_items, _num_threads);
        const size_t chunk_size = (work_items + task_count - 1) / task_count;

    // Use atomics for coordination to avoid capturing references to
    // local mutexes/condition_variables inside tasks. Capturing local
    // mutexes/condition_variables by-reference and then executing the
    // tasks asynchronously can lead to situations where a task tries
    // to lock a destroyed mutex (for example during shutdown), which
    // results in `std::system_error: mutex lock failed: Invalid
    // argument`.
    std::shared_ptr<std::atomic_size_t> remaining = std::make_shared<std::atomic_size_t>(task_count);

    // Create shared synchronization primitives on the heap so tasks can
    // safely notify the parent without capturing stack locals.
    auto done_mutex = std::make_shared<std::mutex>();
    auto done_cv = std::make_shared<std::condition_variable>();

    // Store the first exception as a pointer stored atomically. We
    // allocate the exception_ptr on the heap and install the pointer
    // with CAS so we don't need to lock local mutexes from tasks.
    std::atomic<void*> first_exception_ptr{nullptr};

    // To avoid allocating a callable per task, create a shared holder for
    // the user-provided callable. Move the callable into a shared object
    // once and have tasks invoke it through a small type-erased wrapper.
    using FnDecay = std::decay_t<F>;
    auto fn_holder = std::make_shared<FnDecay>(std::forward<F>(fn));
    auto callable = std::make_shared<std::function<void(size_t)>>( [fn_holder](size_t i) { (*fn_holder)(i); } );

        for(size_t task_index = 0; task_index < task_count; ++task_index) {
            const size_t chunk_begin = begin + task_index * chunk_size;
            const size_t chunk_end = std::min(end, chunk_begin + chunk_size);

            if(chunk_begin >= chunk_end) {
                // Nothing to do for an empty chunk: decrement remaining and
                // if this was the last chunk notify the parent via the
                // shared condition variable.
                if(remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(*done_mutex);
                    done_cv->notify_one();
                }
                continue;
            }

            // Build the task as a std::function so we can either enqueue it or
            // run it inline if the runtime is stopping. Use shared_ptr/atomics
            // for synchronization so tasks do not reference local mutexes.
            auto remaining_sp = remaining;
            std::function<void()> task = [chunk_begin,
                                          chunk_end,
                                          callable,
                                          remaining_sp,
                                          done_mutex,
                                          done_cv,
                                          &first_exception_ptr]() mutable {
                try {
                    for(size_t i = chunk_begin; i < chunk_end; ++i) {
                        (*callable)(i);
                    }
                } catch(...) {
                    // Capture the exception_ptr on the heap and try to
                    // install it as the first exception using CAS.
                    try {
                        auto ep = std::make_unique<std::exception_ptr>(std::current_exception());
                        void* expected = nullptr;
                        void* desired = ep.get();
                        if(first_exception_ptr.compare_exchange_strong(expected, desired, std::memory_order_acq_rel)) {
                            // We successfully stored the pointer; release ownership
                            // from the unique_ptr so it stays alive until we read it
                            // back in the parent thread.
                            (void)ep.release();
                        }
                        // If we didn't install it, someone else already did and
                        // will own the pointer.
                    } catch(...) {
                        // If allocating the exception_ptr fails, there's not much
                        // we can do here.
                    }
                }

                if(remaining_sp->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(*done_mutex);
                    done_cv->notify_one();
                }
            };

            // Try to enqueue - if enqueue fails because the runtime is
            // stopping, execute the task inline to avoid throwing and
            // unwinding while other tasks might still reference local mutexes.
            if(!enqueue(task)) {
                task();
            }
        }


        // Wait for all tasks to finish using the shared condition variable
        // created above. This avoids busy-waiting while still using heap
        // allocated synchronization primitives that tasks can safely
        // reference.
        std::unique_lock<std::mutex> lk(*done_mutex);
        done_cv->wait(lk, [&remaining]() { return remaining->load(std::memory_order_acquire) == 0; });

        // If an exception pointer was installed, take ownership and rethrow.
        void* eptr = first_exception_ptr.load(std::memory_order_acquire);
        if(eptr != nullptr) {
            std::exception_ptr* ep = static_cast<std::exception_ptr*>(eptr);
            std::exception_ptr ex = *ep;
            // Clean up the heap-allocated exception_ptr
            delete ep;
            std::rethrow_exception(ex);
        }
    }

private:
    void startWorkers();
    void stopWorkers();
    void workerLoop();

};
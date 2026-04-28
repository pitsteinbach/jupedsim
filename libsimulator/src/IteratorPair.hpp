// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

/// Pair of Iterators for range based for loops
///
/// Provides begin and end method required for range based for loops.
/// _it_second must be reachable by _it_first.
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
template <typename IteratorFirst, typename IteratorSecond = IteratorFirst>
class IteratorPair
{
    IteratorFirst _it_first;
    IteratorSecond _it_second;

public:
    IteratorPair(IteratorFirst it1, IteratorSecond it2) : _it_first(it1), _it_second(it2) {}

    IteratorFirst first() const { return _it_first; }
    IteratorSecond second() const { return _it_second; }

    IteratorFirst begin() const { return first(); }
    IteratorSecond end() const { return second(); }

    bool empty() const { return _it_first == _it_second; }
    size_t size() const { return std::distance(_it_first, _it_second); }

    // Provide indexing to allow existing code to use agents[idx]. This will
    // work for random-access iterators in O(1) and for weaker iterators in
    // O(n) (std::advance).
    auto operator[](size_t idx) const -> decltype(*std::declval<IteratorFirst>())
    {
        IteratorFirst it = _it_first;
        std::advance(it, static_cast<std::ptrdiff_t>(idx));
        return *it;
    }

    // Bounds-checked access; throws std::out_of_range if idx >= size().
    auto at(size_t idx) const -> decltype(*std::declval<IteratorFirst>())
    {
        if(idx >= size()) {
            throw std::out_of_range("IteratorPair::at: index out of range");
        }
        return operator[](idx);
    }
};

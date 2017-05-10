// -*- C++ -*-

#include "stlib/ads/priority_queue/PriorityQueueBinaryHeapArray.h"

#include "stlib/ads/array/Array.h"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  using namespace ads;

  // The number type.
  typedef double number_type;
  // An array of numbers.
  typedef Array< 2, number_type > number_array_type;
  // The size of a multi-array.
  typedef number_array_type::index_type index_type;

  // The element type to store in the heap.
  typedef number_array_type::const_iterator element_type;

  // The priority queue.
  typedef PriorityQueueBinaryHeapArray<element_type> pq_type;

  {
    number_array_type solution(index_type(4, 4), 100);

    pq_type pq(solution);

    assert(pq.empty() && pq.size() == 0);

    // Add the data to the priority queue.
    for (int i = solution.lbound(0); i != solution.ubound(0); ++i) {
      for (int j = solution.lbound(1); j != solution.ubound(1); ++j) {
        pq.push(&solution(i, j));
      }
    }
    assert(! pq.empty() && pq.size() == solution.size());

    // Decrease the keys.
    for (int i = solution.lbound(0); i != solution.ubound(0); ++i) {
      for (int j = solution.lbound(1); j != solution.ubound(1); ++j) {
        solution(i, j) = i + j;
        pq.decrease(&solution(i, j));
      }
    }
    // Print the sorted values.
    std::vector<double> sorted_values;
    while (! pq.empty()) {
      std::cout << *pq.top() << '\n';
      sorted_values.push_back(*pq.top());
      pq.pop();
    }
    assert(std::is_sorted(sorted_values.begin(), sorted_values.end()));
  }

  return 0;
}

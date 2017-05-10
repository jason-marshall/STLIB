// -*- C++ -*-

#include "stlib/ads/priority_queue/PriorityQueueBinaryHeapDynamicKeys.h"

#include "stlib/ads/priority_queue/HeapHandleArray.h"

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
  typedef Array<2, number_type> number_array_type;
  // The size of a multi-array.
  typedef number_array_type::index_type index_type;

  // The element type to store in the heap.
  typedef number_array_type::const_iterator element_type;
  // A handle into the heap.
  typedef PriorityQueue<element_type>::iterator
  heap_handle;

  // An array of heap handles.
  typedef Array< 2, heap_handle > heap_handle_array_type;

  // The functor to get handles.
  typedef HeapHandleArray< element_type, heap_handle >
  get_handle_functor;

  // The priority queue.
  typedef PriorityQueueBinaryHeapDynamicKeys < element_type,
          get_handle_functor > pq_type;

  {
    number_array_type solution(index_type(4, 4), 100);
    heap_handle_array_type heap_handles(index_type(4, 4));

    get_handle_functor get_handle(solution, heap_handles);
    pq_type pq(get_handle);

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

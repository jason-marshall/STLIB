// -*- C++ -*-

#include "stlib/ads/priority_queue/PriorityQueueBinaryHeapStoreKeys.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;
  {
    typedef PriorityQueueBinaryHeapStoreKeys< const int* > pq_type;
    const int size = 10;
    // The data.
    int data[size] = {23, 2, 7, 11, 3, 5, 29, 13, 17, 19};
    pq_type pq;
    assert(pq.empty() && pq.size() == 0);

    // Add the data to the priority queue.
    for (int i = 0; i != size; ++i) {
      pq.push(data + i);
    }
    assert(! pq.empty() && int(pq.size()) == size);

    // Print the sorted values.
    std::vector<int> sorted_values;
    while (! pq.empty()) {
      std::cout << *pq.top() << '\n';
      sorted_values.push_back(*pq.top());
      pq.pop();
    }
    std::cout << '\n';
    assert(std::is_sorted(sorted_values.begin(), sorted_values.end()));
    sorted_values.clear();

    // Add the data to the priority queue.  Specify the keys.
    for (int i = 0; i != size; ++i) {
      pq.push(data + i, data[i]);
    }
    assert(! pq.empty() && int(pq.size()) == size);

    // Print the sorted values.
    while (! pq.empty()) {
      std::cout << *pq.top() << '\n';
      sorted_values.push_back(*pq.top());
      pq.pop();
    }
    std::cout << '\n';
    assert(std::is_sorted(sorted_values.begin(), sorted_values.end()));
    sorted_values.clear();
  }

  return 0;
}

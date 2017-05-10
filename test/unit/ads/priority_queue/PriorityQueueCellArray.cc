// -*- C++ -*-

#include "stlib/ads/priority_queue/PriorityQueueCellArray.h"

#include <iostream>
#include <list>

using namespace stlib;

int
main()
{
  using namespace ads;
  {
    typedef std::list<double> data_container_type;
    typedef data_container_type::const_iterator element_type;

    typedef PriorityQueueCellArray< element_type > priority_queue;
    typedef priority_queue::container_type container_type;
    typedef container_type::const_iterator const_iterator;

    const int size = 10;
    data_container_type data;
    // The data.
    for (int i = 0; i != size; ++i) {
      data.push_back(0.1 * i);
    }
    priority_queue pq(0, 0.1, 1);
    assert(pq.empty() && pq.size() == 0);

    // Add the elements to the priority queue.
    for (element_type i = data.begin(); i != data.end(); ++i) {
      pq.push(i);
    }
    assert(! pq.empty() && int(pq.size()) == size);

    // Take some steps popping and pushing.
    const int num_steps = 10;
    for (int i = 0; i != num_steps; ++i) {
      for (const_iterator iter = pq.top().begin(); iter != pq.top().end();
           ++iter) {
        std::cout << **iter << '\n';
        pq.push(data.insert(data.end(), **iter + 1));
      }
      pq.pop();
    }
    // Pop until the priority queue is empty.
    while (! pq.empty()) {
      for (const_iterator iter = pq.top().begin(); iter != pq.top().end();
           ++iter) {
        std::cout << **iter << '\n';
      }
      pq.pop();
    }
  }

  return 0;
}

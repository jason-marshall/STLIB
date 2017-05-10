// -*- C++ -*-

#include "stlib/shortest_paths/SortedHeap.h"

#include "stlib/shortest_paths/VertexDijkstra.h"
#include "stlib/shortest_paths/VertexCompare.h"

#include <iostream>
#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  typedef VertexDijkstra<double> VertexType;
  typedef SortedHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

  {
    HeapType heap;

    std::vector<VertexType> graph;
    const int size = 100;

    for (int i = 0; i < size; ++i) {
      VertexType vertex;
      vertex.set_distance(i);
      graph.push_back(vertex);
    }

    for (int i = 0; i < size; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 0; i < size; ++i) {
      assert(heap.top()->distance() == i);
      heap.pop();
    }

    for (int i = size - 1; i >= 0; --i) {
      heap.push(&graph[i]);
    }

    for (int i = 0; i < size; ++i) {
      assert(heap.top()->distance() == i);
      heap.pop();
    }
  }

  {
    HeapType heap;

    std::vector<VertexType> graph;
    const int size = 100;

    for (int i = 0; i < size; ++i) {
      VertexType vertex;
      vertex.set_distance(1000);
      graph.push_back(vertex);
    }

    for (int i = 0; i < size; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 0; i < size; ++i) {
      graph[i].set_distance(size - i - 1);
      heap.decrease(graph[i].heap_ptr());
    }

    for (int i = 0; i < size; ++i) {
      assert(heap.top()->distance() == i);
      heap.pop();
    }
  }

  return 0;
}

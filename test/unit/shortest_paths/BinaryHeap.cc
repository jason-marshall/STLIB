// -*- C++ -*-

#include "stlib/shortest_paths/BinaryHeap.h"

#include "stlib/shortest_paths/VertexDijkstra.h"
#include "stlib/shortest_paths/VertexCompare.h"

#include <iostream>
#include <algorithm>
#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    typedef VertexDijkstra<double> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;

    {
      VertexType vertex;
      vertex.set_distance(0);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(1);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(2);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(3);
      graph.push_back(vertex);
    }

    heap.push(&graph[0]);
    heap.push(&graph[1]);
    heap.push(&graph[2]);
    heap.push(&graph[3]);

    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << '\n';

    heap.push(&graph[3]);
    heap.push(&graph[2]);
    heap.push(&graph[1]);
    heap.push(&graph[0]);

    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << '\n';
  }

  {
    typedef VertexDijkstra<double> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;
    VertexType vertex;

    {
      VertexType vertex;
      vertex.set_distance(10);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(10);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(10);
      graph.push_back(vertex);
    }
    {
      VertexType vertex;
      vertex.set_distance(10);
      graph.push_back(vertex);
    }

    heap.push(&graph[0]);
    heap.push(&graph[1]);
    heap.push(&graph[2]);
    heap.push(&graph[3]);

    graph[0].set_distance(3);
    heap.decrease(graph[0].heap_ptr());
    graph[1].set_distance(2);
    heap.decrease(graph[1].heap_ptr());
    graph[2].set_distance(1);
    heap.decrease(graph[2].heap_ptr());
    graph[3].set_distance(0);
    heap.decrease(graph[3].heap_ptr());

    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << *heap.top() << '\n';
    heap.pop();
    std::cout << '\n';
  }

  {
    typedef VertexDijkstra<int> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;
    VertexType vertex;

    for (int i = 0; i < 1000; ++i) {
      VertexType vertex;
      vertex.set_distance(i);
      graph.push_back(vertex);
    }

    std::random_shuffle(graph.begin(), graph.end());

    for (int i = 0; i < 1000; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 0; i < 1000; ++i) {
      assert(heap.top()->distance() == i);
      heap.pop();
    }
  }

  {
    typedef VertexDijkstra<int> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;
    VertexType vertex;

    for (int i = 0; i < 1000; ++i) {
      VertexType vertex;
      vertex.set_distance(10000);
      graph.push_back(vertex);
    }

    std::random_shuffle(graph.begin(), graph.end());

    for (int i = 0; i < 1000; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 999; i >= 0; --i) {
      graph[i].set_distance(i);
      heap.decrease(graph[i].heap_ptr());
    }

    for (int i = 0; i < 1000; ++i) {
      assert(heap.top()->distance() == i);
      heap.pop();
    }
  }

  {
    typedef VertexDijkstra<int> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;
    VertexType vertex;

    for (int i = 0; i < 1000; ++i) {
      VertexType vertex;
      vertex.set_distance(10000);
      graph.push_back(vertex);
    }

    std::random_shuffle(graph.begin(), graph.end());

    for (int i = 0; i < 1000; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 999; i >= 0; --i) {
      graph[i].set_distance(i / 10);
      heap.decrease(graph[i].heap_ptr());
    }

    for (int i = 0; i < 1000; ++i) {
      assert(heap.top()->distance() == i / 10);
      heap.pop();
    }
  }

  {
    typedef VertexDijkstra<int> VertexType;
    typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;

    HeapType heap;

    std::vector<VertexType> graph;
    VertexType vertex;

    for (int i = 0; i < 1000; ++i) {
      VertexType vertex;
      vertex.set_distance(10000);
      graph.push_back(vertex);
    }

    std::random_shuffle(graph.begin(), graph.end());

    for (int i = 0; i < 1000; ++i) {
      heap.push(&graph[i]);
    }

    for (int i = 0; i < 1000; ++i) {
      graph[i].set_distance(i / 10);
      heap.decrease(graph[i].heap_ptr());
    }

    for (int i = 0; i < 1000; ++i) {
      assert(heap.top()->distance() == i / 10);
      heap.pop();
    }
  }

  return 0;
}

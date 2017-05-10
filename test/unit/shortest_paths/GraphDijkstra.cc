// -*- C++ -*-

#include "stlib/shortest_paths/GraphDijkstra.h"
#include "stlib/shortest_paths/BinaryHeap.h"
#include "stlib/shortest_paths/UniformRandom.h"
#include "stlib/shortest_paths/GraphBellmanFord.h"

#include "stlib/ads/functor/constant.h"

#include <iostream>
#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  typedef VertexDijkstra<double> VertexType;
  typedef BinaryHeap< VertexType*, VertexCompare<VertexType*> > HeapType;
  typedef GraphDijkstra<double, HeapType> GraphType;

  {
    GraphType graph;
    UniformRandom<double> edge_weight(0);
    graph.rectangular_grid(3, 3, edge_weight);
    graph.dijkstra(0);
    std::cout << graph << '\n';
  }

  {
    GraphType graph;
    ads::GeneratorConstant<double> edge_weight(1.);
    graph.rectangular_grid(100, 100, edge_weight);
    graph.dijkstra(0);

    GraphBellmanFord<double> graph_bf;
    graph_bf.rectangular_grid(100, 100, edge_weight);
    graph_bf.bellman_ford(0);

    std::cout << graph << '\n' << graph_bf;
    assert(graph == graph_bf);
  }

  return 0;
}

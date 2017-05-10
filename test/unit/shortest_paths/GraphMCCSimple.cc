// -*- C++ -*-

#include "stlib/shortest_paths/GraphMCCSimple.h"
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

  typedef GraphMCCSimple<double> GraphType;
  {
    GraphType graph;

    UniformRandom<double> edge_weight(0);
    graph.rectangular_grid(3, 3, edge_weight);
    graph.marching_with_correctness_criterion(0);
    std::cout << graph << '\n';
  }

  {
    GraphType graph;

    ads::GeneratorConstant<double> edge_weight(1.);
    graph.rectangular_grid(10, 10, edge_weight);
    graph.marching_with_correctness_criterion(0);

    GraphBellmanFord<double> graph_bf;
    graph_bf.rectangular_grid(10, 10, edge_weight);
    graph_bf.bellman_ford(0);

    assert(graph == graph_bf);
  }

  {
    GraphType graph;

    ads::GeneratorConstant<double> edge_weight(1.);
    graph.rectangular_grid(100, 100, edge_weight);
    graph.marching_with_correctness_criterion(0);

    GraphBellmanFord<int> graph_bf;
    graph_bf.rectangular_grid(100, 100, edge_weight);
    graph_bf.bellman_ford(0);

    assert(graph == graph_bf);
  }

  return 0;
}

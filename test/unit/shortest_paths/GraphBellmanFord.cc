// -*- C++ -*-

#include "stlib/shortest_paths/GraphBellmanFord.h"

#include "stlib/shortest_paths/UniformRandom.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    typedef GraphBellmanFord< double > GraphType;
    GraphType graph;

    UniformRandom<double> edge_weight(0);
    graph.rectangular_grid(3, 3, edge_weight);
    graph.bellman_ford(0);
    std::cout << graph << '\n';
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/shortest_paths/Graph.h"

#include "stlib/shortest_paths/Vertex.h"
#include "stlib/shortest_paths/UniformRandom.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    typedef Graph< Vertex<double> > GraphType;
    GraphType graph;

    UniformRandom<double> edge_weight(0);
    graph.rectangular_grid(2, 3, edge_weight);
    std::cout << "A 2x3 rectangular grid." << '\n'
              << graph << '\n';

    graph.dense(4, edge_weight);
    std::cout << "A dense graph with 4 vertices." << '\n'
              << graph << '\n';
  }

  return 0;
}

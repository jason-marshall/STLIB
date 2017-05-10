// -*- C++ -*-

#include "stlib/shortest_paths/HalfEdge.h"

#include "stlib/shortest_paths/Vertex.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    // Default constructor
    HalfEdge< Vertex<int> > edge;
  }
  {
    Vertex<double> vertex;
    HalfEdge< Vertex<double> > edge(&vertex, 42);
    assert(edge.vertex() == &vertex);
    assert(edge.weight() == 42);

    std::cout << edge << '\n';
  }
  return 0;
}

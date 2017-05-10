// -*- C++ -*-

#include "stlib/shortest_paths/Edge.h"

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
    Edge< Vertex<int> > edge;
  }
  {
    Vertex<double> source, target;
    Edge< Vertex<double> > edge(&source, &target, 42);
    assert(edge.source() == &source);
    assert(edge.target() == &target);
    assert(edge.weight() == 42);

    std::cout << edge << '\n';
  }
  return 0;
}

// -*- C++ -*-

#include "stlib/shortest_paths/Vertex.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    // Default constructor
    Vertex<int> vertex;
  }
  {
    Vertex<double> vertex;
    vertex.initialize();
    assert(vertex.distance() == std::numeric_limits<double>::max() &&
           vertex.predecessor() == 0);
    vertex.set_root();
    assert(vertex.distance() == 0 && vertex.predecessor() == 0);

    std::cout << vertex << '\n';
  }
  return 0;
}

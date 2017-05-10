// -*- C++ -*-

#include "stlib/shortest_paths/VertexMCCSimple.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    VertexMCCSimple<double> vertex;
    vertex.initialize();
    assert(vertex.status() == UNLABELED);
    vertex.set_root();
    assert(vertex.status() == KNOWN && vertex.distance() == 0 &&
           vertex.predecessor() == 0);

    std::cout << vertex << '\n';
  }
  {
    VertexMCCSimple<double> v0;
    v0.set_root();
    VertexMCCSimple<double> v1;
    v1.initialize();
    v1.label(v0, 1);
    assert(v1.distance() == 1);
  }
  return 0;
}

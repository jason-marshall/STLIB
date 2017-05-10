// -*- C++ -*-

#include "stlib/shortest_paths/VertexLabel.h"

#include <iostream>
#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  {
    VertexLabel<double> vertex;
    vertex.initialize();
    assert(vertex.status() == UNLABELED);
    vertex.set_root();
    assert(vertex.status() == KNOWN && vertex.distance() == 0 &&
           vertex.predecessor() == 0);

    std::cout << vertex << '\n';
  }
  {
    VertexLabel<double> v0;
    v0.set_root();
    VertexLabel<double> v1;
    v1.initialize();
    v1.label(v0, 2);
    assert(v1.distance() == 2);
    assert(v1.status() == LABELED);
    v1.label(v0, 3);
    assert(v1.distance() == 2);
    assert(v1.status() == LABELED);
    v1.label(v0, 1);
    assert(v1.distance() == 1);
    assert(v1.status() == LABELED);
  }
  return 0;
}

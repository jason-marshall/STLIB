// -*- C++ -*-

#include "stlib/shortest_paths/VertexMCC.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  typedef VertexMCC<double> VertexType;

  {
    VertexType vertex;
    vertex.initialize();
    assert(vertex.status() == UNLABELED);
    vertex.set_root();
    assert(vertex.status() == KNOWN && vertex.distance() == 0 &&
           vertex.predecessor() == 0);

    std::cout << vertex << '\n';

    VertexType copy(vertex);
    assert(vertex == copy);
    VertexType assign;
    assign = vertex;
    assert(vertex == assign);
  }
  return 0;
}

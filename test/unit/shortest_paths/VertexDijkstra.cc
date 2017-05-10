// -*- C++ -*-

#include "stlib/shortest_paths/VertexDijkstra.h"

#include <iostream>
#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  typedef VertexDijkstra<double> VertexType;
  typedef HalfEdge< VertexType > EdgeType;

  {
    VertexType vertex;
    vertex.initialize();
    assert(vertex.status() == UNLABELED);
    vertex.set_root();
    assert(vertex.status() == KNOWN && vertex.distance() == 0 &&
           vertex.predecessor() == 0);

    std::cout << vertex << '\n';
  }
  {
    VertexType v0;
    v0.set_root();
    VertexType v1;
    v1.initialize();
    EdgeType e(&v1, 1);
    v0.set_adjacent_edges(&e);
    assert(v0.adjacent_edges() == &e);
    assert(v1.adjacent_edges() == 0);
  }
  return 0;
}

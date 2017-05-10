// -*- C++ -*-

#include "stlib/shortest_paths/VertexCompare.h"

#include "stlib/shortest_paths/Vertex.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace shortest_paths;

  typedef Vertex<double> VertexType;

  {
    Vertex<double> a;
    a.set_distance(0);
    Vertex<double> b;
    b.set_distance(1);

    {
      VertexCompare<VertexType> vc;
      assert(vc(a, b));
    }
    {
      VertexCompareGreater<VertexType> vc;
      assert(vc(b, a));
    }
    {
      VertexCompare<VertexType*> vc;
      assert(vc(&a, &b));
    }
    {
      VertexCompareGreater<VertexType*> vc;
      assert(vc(&b, &a));
    }
  }
  return 0;
}

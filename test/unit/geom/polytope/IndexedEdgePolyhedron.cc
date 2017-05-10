// -*- C++ -*-

#include "stlib/geom/polytope/IndexedEdgePolyhedron.h"

#include <iostream>

#include <cassert>

using namespace stlib;
using namespace geom;

int
main()
{
  typedef IndexedEdgePolyhedron<double> Polyhedron;
  typedef Polyhedron::Point Point;
  typedef BBox<double, 3> BBox;

  {
    // Default constructor.
    Polyhedron x;
    assert(x.getVerticesSize() == 0 && x.getEdgesSize() == 0);
  }
  {
    Polyhedron x;

    //
    // Make a cube.
    //

    x.insertVertex(Point{{0, 0, 0}});
    x.insertVertex(Point{{1, 0, 0}});
    x.insertVertex(Point{{1, 1, 0}});
    x.insertVertex(Point{{0, 1, 0}});

    x.insertVertex(Point{{0, 0, 1}});
    x.insertVertex(Point{{1, 0, 1}});
    x.insertVertex(Point{{1, 1, 1}});
    x.insertVertex(Point{{0, 1, 1}});

    assert(x.getVerticesSize() == 8);

    x.insertEdge(0, 1);
    x.insertEdge(1, 2);
    x.insertEdge(2, 3);
    x.insertEdge(3, 0);

    x.insertEdge(0, 4);
    x.insertEdge(1, 5);
    x.insertEdge(2, 6);
    x.insertEdge(3, 7);

    x.insertEdge(4, 5);
    x.insertEdge(5, 6);
    x.insertEdge(6, 7);
    x.insertEdge(7, 4);

    assert(x.getEdgesSize() == 12);
    assert(x.getEdgeSource(0) == (Point{{0, 0, 0}}));
    assert(x.getEdgeTarget(0) == (Point{{1, 0, 0}}));

    {
      // Copy constructor.
      Polyhedron y(x);
      assert(x == y);
    }

    {
      // Assignment operator.
      Polyhedron y;
      assert(x != y);
      y = x;
      assert(x == y);
    }

    // Bounding box.
    BBox bb;
    x.computeBBox(&bb);
    assert(bb == (BBox{Point{{0, 0, 0}}, Point{{1, 1 , 1}}}));

    x.clear();
    assert(x.getVerticesSize() == 0 && x.getEdgesSize() == 0);
  }

  return 0;
}

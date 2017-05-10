// -*- C++ -*-

#include "stlib/geom/polytope/ScanConversionPolyhedron.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::ScanConversionPolyhedron<std::ptrdiff_t, double> Polyhedron;
  typedef Polyhedron::Point Point;
  typedef geom::BBox<double, 3> BB;
  typedef geom::RegularGrid<3> RegularGrid;
  typedef RegularGrid::SizeList SizeList;
  typedef std::array<std::ptrdiff_t, 3> IndexList;

  {
    // default constructor
    std::cout << "ScanConversionPolyhedron<double>() = " << '\n'
              << Polyhedron();
  }
  {
    //
    // make a cube
    //
    Polyhedron polyhedron;
    // bottom
    polyhedron.insertEdge(Point{{0, 0, 0}}, Point{{1, 0, 0}});
    polyhedron.insertEdge(Point{{1, 0, 0}}, Point{{1, 1, 0}});
    polyhedron.insertEdge(Point{{1, 1, 0}}, Point{{0, 1, 0}});
    polyhedron.insertEdge(Point{{0, 1, 0}}, Point{{0, 0, 0}});
    //top
    polyhedron.insertEdge(Point{{0, 0, 1}}, Point{{1, 0, 1}});
    polyhedron.insertEdge(Point{{1, 0, 1}}, Point{{1, 1, 1}});
    polyhedron.insertEdge(Point{{1, 1, 1}}, Point{{0, 1, 1}});
    polyhedron.insertEdge(Point{{0, 1, 1}}, Point{{0, 0, 1}});
    // sides
    polyhedron.insertEdge(Point{{0, 0, 0}}, Point{{0, 0, 1}});
    polyhedron.insertEdge(Point{{1, 0, 0}}, Point{{1, 0, 1}});
    polyhedron.insertEdge(Point{{1, 1, 0}}, Point{{1, 1, 1}});
    polyhedron.insertEdge(Point{{0, 1, 0}}, Point{{0, 1, 1}});

    //
    // print the cube
    //
    std::cout << '\n' << "cube = " << '\n'
              << polyhedron;

    // copy constructor
    Polyhedron copy(polyhedron);
    assert(copy == polyhedron);

    // assignment operator
    Polyhedron other;
    other = polyhedron;
    assert(other == polyhedron);

    // Make a bbox
    BB bb;
    polyhedron.computeBBox(&bb);
    assert((bb == BB{Point{{0, 0, 0}}, Point{{1, 1, 1}}}));

    // scan convert
    BB domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    RegularGrid grid(SizeList{{2, 2, 2}}, domain);
    std::vector<IndexList> cs;
    polyhedron.scanConvert(std::back_inserter(cs), grid);
    assert(cs.size() == 8);
  }
  {
    //
    // Assign from an indexed edge polyhedron.

    geom::IndexedEdgePolyhedron<double> x;

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

    Polyhedron y;
    y = x;
    assert(y.getEdges().size() == 4);
    geom::BBox<double, 3> box;
    y.computeBBox(&box);
    geom::BBox<double, 3> unit = {{{0, 0, 0}}, {{1, 1 , 1}}};
    assert(box == unit);
  }
  {
    //
    // make a prizm
    //
    Polyhedron polyhedron;
    // sides
    polyhedron.insertEdge(Point{{-0.2, -0.1 , -0.1}}, Point{{-0.2, -0.1, 10.1}});
    polyhedron.insertEdge(Point{{10.1, -0.1 , -0.1}}, Point{{10.1, -0.1, 10.1}});
    polyhedron.insertEdge(Point{{10.1, 10.2, -0.1}}, Point{{10.1, 10.2, 10.1}});

    // Make a bbox
    BB bb;
    polyhedron.computeBBox(&bb);
    assert((bb == BB{Point{{-0.2, -0.1, -0.1}}, Point{{10.1, 10.2, 10.1}}}));

    // scan convert
    BB domain = {{{0, 0, 0}}, {{10, 10, 10}}};
    RegularGrid grid(SizeList{{10, 10, 10}}, domain);
    std::vector<IndexList> cs;
    polyhedron.scanConvert(std::back_inserter(cs), grid);
    assert(cs.size() == 550);
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/geom/polytope/ScanConversionPolygon.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::ScanConversionPolygon<std::ptrdiff_t, double> Polygon;
  typedef Polygon::Point Point;
  typedef std::array<std::ptrdiff_t, 2> Index2;
  typedef std::array<std::ptrdiff_t, 3> Index3;
  typedef geom::RegularGrid<3> RegularGrid3;
  typedef geom::RegularGrid<2> RegularGrid2;
  typedef RegularGrid3::SizeList SizeList3;
  typedef RegularGrid2::SizeList SizeList2;
  typedef geom::BBox<double, 3> BBox3;
  typedef geom::BBox<double, 2> BBox2;
  typedef geom::Line_2<double> Line;
  {
    // default constructor
    std::cout << "ScanConversionPolygon() = " << '\n'
              << Polygon();
  }
  {
    // size constructor
    std::cout << "Polygon(2) = " << '\n'
              << Polygon(2);
  }
  {
    // Equality
    Polygon a, b;
    assert(a == b);
    a.insert(Point{{0, 0}});
    b.insert(Point{{0, 0}});
    assert(a == b);
    a.insert(Point{{1, 2}});
    b.insert(Point{{1, 2}});
    assert(a == b);
  }
  {
    // Inequality
    Polygon a, b;
    assert(!(a != b));
    a.insert(Point{{0, 0}});
    assert(a != b);
    b.insert(Point{{1, 0}});
    assert(a != b);
    a.insert(Point{{1, 2}});
    b.insert(Point{{1, 2}});
    assert(a != b);
  }
  {
    // Copy constructor.
    Polygon a;
    a.insert(Point{{0, 0}});
    a.insert(Point{{1, 2}});
    Polygon b(a);
    assert(a == b);
  }
  {
    // Assignment operator.
    Polygon a;
    a.insert(Point{{0, 0}});
    a.insert(Point{{1, 2}});
    Polygon b = a;
    assert(a == b);
  }
  {
    // order_vertices
    Polygon a;
    a.insert(Point{{0, 0}});
    a.insert(Point{{1, 1}});
    a.insert(Point{{0, 1}});
    a.insert(Point{{1, 0}});
    a.orderVertices();
    Polygon b;
    b.insert(Point{{0, 0}});
    b.insert(Point{{1, 0}});
    b.insert(Point{{1, 1}});
    b.insert(Point{{0, 1}});
    assert(a == b);
  }
  {
    // remove_duplicates
    Polygon a;
    a.insert(Point{{0, 0}});
    a.insert(Point{{1, 0}});
    a.insert(Point{{1, 0}});
    a.insert(Point{{1, 1}});
    a.insert(Point{{0, 1}});
    a.insert(Point{{0, 1}});
    a.insert(Point{{0, 1}});
    a.insert(Point{{0, 1}});
    a.removeDuplicates();

    Polygon b;
    b.insert(Point{{0, 0}});
    b.insert(Point{{1, 0}});
    b.insert(Point{{1, 1}});
    b.insert(Point{{0, 1}});
    assert(a == b);
  }
  {
    // bottom_and_top
    Polygon a;
    a.insert(Point{{0, 0}});
    a.insert(Point{{1, 1}});
    a.insert(Point{{0, 1}});
    a.insert(Point{{1, 0}});
    double top, bottom;
    assert(a.computeBottomAndTop(&bottom, &top) == 0);
    assert(bottom == 0 && top == 1);
  }
  {
    // is_valid
    Polygon a;
    assert(! a.isValid());
    a.insert(Point{{0, 0}});
    assert(! a.isValid());
    a.insert(Point{{1, 1}});
    assert(! a.isValid());
    a.insert(Point{{0, 1}});
    assert(a.isValid());
    a.insert(Point{{1, 0}});
    assert(a.isValid());
    a.insert(Point{{1, 0}});
    assert(! a.isValid());
  }
  {
    // scan_convert 2-D
    std::vector<Index2> cs;
    BBox2 domain = {{{0, 0}}, {{1, 1}}};
    RegularGrid2 grid(SizeList2{{11, 11}}, domain);

    Polygon p;

    // Point, degenerate.
    p.insert(Point{{-0.2, -0.1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 0);

    // Line segment, degenerate.
    p.insert(Point{{10.1, -0.1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 0);

    // Triangle.
    p.insert(Point{{10.1, 10.2}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 66);

    // Square.
    p.insert(Point{{-0.2, 10.2}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 121);

    // Big triangle.
    p.clear();
    p.insert(Point{{-30, -10}});
    p.insert(Point{{30, -10}});
    p.insert(Point{{30, 50}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 121);

    // Triangle with mid-points.
    p.clear();
    p.insert(Point{{-0.2, -0.1}});
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 10.2}});
    p.insert(Point{{5, 5.2}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    // CONTINUE REMOVE
#if 0
    std::cerr << cs.size() << "\n";
    for (std::vector<Index2>::const_iterator i = cs.begin(); i != cs.end();
         ++i)  {
      std::cerr << *i << "\n";
    }
#endif
    assert(cs.size() == 66);

    // Triangle with mid-points.
    p.clear();
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 10.2}});
    p.insert(Point{{5, 5.2}});
    p.insert(Point{{-0.2, -0.1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 66);

    // Triangle with mid-points.
    p.clear();
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 10.2}});
    p.insert(Point{{5, 5.2}});
    p.insert(Point{{-0.2, -0.1}});
    p.insert(Point{{5, -0.1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 66);

    // Triangle with many points.
    p.clear();
    p.insert(Point{{-0.2, -0.1}});
    p.insert(Point{{1, -0.1}});
    p.insert(Point{{2, -0.1}});
    p.insert(Point{{3, -0.1}});
    p.insert(Point{{4, -0.1}});
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{6, -0.1}});
    p.insert(Point{{7, -0.1}});
    p.insert(Point{{8, -0.1}});
    p.insert(Point{{9, -0.1}});
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 1}});
    p.insert(Point{{10.1, 2}});
    p.insert(Point{{10.1, 3}});
    p.insert(Point{{10.1, 4}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 6}});
    p.insert(Point{{10.1, 7}});
    p.insert(Point{{10.1, 8}});
    p.insert(Point{{10.1, 9}});
    p.insert(Point{{10.1, 10.2}});
    p.insert(Point{{9, 9.2}});
    p.insert(Point{{8, 8.2}});
    p.insert(Point{{7, 7.2}});
    p.insert(Point{{6, 6.2}});
    p.insert(Point{{5, 5.2}});
    p.insert(Point{{4, 4.2}});
    p.insert(Point{{3, 3.2}});
    p.insert(Point{{2, 2.2}});
    p.insert(Point{{1, 1.2}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 66);

    // Square with mid-points.
    p.clear();
    p.insert(Point{{-0.1, -0.1}});
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 10.1}});
    p.insert(Point{{5, 10.1}});
    p.insert(Point{{-0.1, 10.1}});
    p.insert(Point{{-0.1, 5}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 121);

    // Square with many points.
    p.clear();
    p.insert(Point{{-0.1, -0.1}});
    p.insert(Point{{1, -0.1}});
    p.insert(Point{{2, -0.1}});
    p.insert(Point{{3, -0.1}});
    p.insert(Point{{4, -0.1}});
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{6, -0.1}});
    p.insert(Point{{7, -0.1}});
    p.insert(Point{{8, -0.1}});
    p.insert(Point{{9, -0.1}});
    p.insert(Point{{10.1, -0.1}});
    p.insert(Point{{10.1, 1}});
    p.insert(Point{{10.1, 2}});
    p.insert(Point{{10.1, 3}});
    p.insert(Point{{10.1, 4}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{10.1, 6}});
    p.insert(Point{{10.1, 7}});
    p.insert(Point{{10.1, 8}});
    p.insert(Point{{10.1, 9}});
    p.insert(Point{{10.1, 10.1}});
    p.insert(Point{{9, 10.1}});
    p.insert(Point{{8, 10.1}});
    p.insert(Point{{7, 10.1}});
    p.insert(Point{{6, 10.1}});
    p.insert(Point{{5, 10.1}});
    p.insert(Point{{4, 10.1}});
    p.insert(Point{{3, 10.1}});
    p.insert(Point{{2, 10.1}});
    p.insert(Point{{1, 10.1}});
    p.insert(Point{{-0.1, 10.1}});
    p.insert(Point{{-0.1, 9}});
    p.insert(Point{{-0.1, 8}});
    p.insert(Point{{-0.1, 7}});
    p.insert(Point{{-0.1, 6}});
    p.insert(Point{{-0.1, 5}});
    p.insert(Point{{-0.1, 4}});
    p.insert(Point{{-0.1, 3}});
    p.insert(Point{{-0.1, 2}});
    p.insert(Point{{-0.1, 1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 121);

    // Diamond.
    p.clear();
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{5, 10.1}});
    p.insert(Point{{-0.1, 5}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 61);

    // Diamond with many points.
    p.clear();
    p.insert(Point{{5, -0.1}});
    p.insert(Point{{6.1, 1}});
    p.insert(Point{{7.1, 2}});
    p.insert(Point{{8.1, 3}});
    p.insert(Point{{9.1, 4}});
    p.insert(Point{{10.1, 5}});
    p.insert(Point{{9, 6.1}});
    p.insert(Point{{8, 7.1}});
    p.insert(Point{{7, 8.1}});
    p.insert(Point{{6, 9.1}});
    p.insert(Point{{5, 10.1}});
    p.insert(Point{{4, 9.1}});
    p.insert(Point{{3, 8.1}});
    p.insert(Point{{2, 7.1}});
    p.insert(Point{{1, 6.1}});
    p.insert(Point{{-0.1, 5}});
    p.insert(Point{{0.9, 4}});
    p.insert(Point{{1.9, 3}});
    p.insert(Point{{2.9, 2}});
    p.insert(Point{{3.9, 1}});
    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents());
    assert(cs.size() == 61);
  }
  {
    // scan_convert 3-D
    std::vector<Index3> cs;
    BBox3 domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    RegularGrid3 grid(SizeList3{{10, 10, 10}}, domain);

    Polygon p;
    p.insert(Point{{-0.2, -0.1}});

    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents(), 0);
    assert(cs.size() == 0);

    p.insert(Point{{9.0, -0.1}});
    p.insert(Point{{9.1, 9.2}});

    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents(), 0);
    assert(cs.size() == 55);

    p.insert(Point{{-0.2, 9.2}});

    cs.clear();
    p.scanConvert(std::back_inserter(cs), grid.getExtents(), 0);
    assert(cs.size() == 100);
  }
  {
    // Clipping.
    Polygon p;
    p.insert(Point{{1, 1}});
    p.insert(Point{{-1, 1}});
    p.insert(Point{{-1, -1}});
    p.insert(Point{{1, -1}});
    {
      // All points above line.
      Line ln(Point{{2, 0}}, Point{{2, -1}});
      Polygon q(p);
      q.clip(ln);
      assert(q == p);
    }
    {
      // All points below line.
      Line ln(Point{{2, 0}}, Point{{2, 1}});
      Polygon q(p);
      q.clip(ln);
      assert(q.getVerticesSize() == 0);
    }
    std::cout << "Polygon:\n" << p;
    {
      // Half points above line.
      Line ln(Point{{1, 0}}, Point{{0, 0}});
      Polygon q(p);
      q.clip(ln);
      std::cout << "Clip by x axis:\n" << q;
      assert(q.getVerticesSize() == 4);
    }
    {
      // Line coincides with edge
      Line ln(Point{{1, -1}}, Point{{0, -1}});
      Polygon q(p);
      q.clip(ln);
      std::cout << "Clip by line coinciding with edge:\n" << q;
      assert(q.getVerticesSize() == 4);
    }
    {
      // Line coincides with vertex.
      Line ln(Point{{1, -1}}, Point{{0, -2}});
      Polygon q(p);
      q.clip(ln);
      std::cout << "Clip by line coinciding with vertex:\n" << q;
      assert(q.getVerticesSize() == 4);
    }
    {
      // Line coincides with vertices
      Line ln(Point{{1, -1}}, Point{{-1, 1}});
      Polygon q(p);
      q.clip(ln);
      std::cout << "Clip by line coinciding with vertices:\n" << q;
      assert(q.getVerticesSize() == 3);
    }
  }
  return 0;
}

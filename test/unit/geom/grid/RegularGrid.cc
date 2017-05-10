// -*- C++ -*-

#include "stlib/geom/grid/RegularGrid.h"

#include "stlib/geom/kernel/content.h"

#include <iostream>

#include <cassert>

using namespace stlib;

using namespace geom;

int
main()
{
  typedef RegularGrid<3> RegularGrid;
  typedef RegularGrid::BBox BBox;
  typedef RegularGrid::SizeList SizeList;
  typedef RegularGrid::Range Range;
  typedef Range::Index Index;
  typedef Range::IndexList IndexList;
  typedef BBox::Point Point;

  //
  // Constructors
  //

  {
    // Default Constructor.
    RegularGrid grid;
  }
  {
    // Construct from grid dimensions and Cartesian domain.
    const SizeList extents = {{2, 2, 2}};
    std::cout << "RegularGrid((2,2,2),BBox(0,0,0,1,1,1)) = " << '\n'
              << RegularGrid(extents,
                             BBox{{{0., 0., 0.}}, {{1., 1., 1.}}})
              << '\n';
  }
  {
    // Copy constructor.
    const SizeList extents = {{2, 2, 2}};
    RegularGrid a(extents,
                  BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    RegularGrid b(a);
    assert(a == b);
  }
  {
    // Assignment operator.
    const SizeList extents = {{2, 2, 2}};
    RegularGrid a(extents,
                  BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    RegularGrid b = a;
    assert(a == b);
  }

  //
  // Accesors: grid size
  //

  {
    const SizeList extents = {{2, 3, 4}};
    RegularGrid grid(extents,
                     BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});

    assert(grid.getExtents()[0] == 2);
    assert(grid.getExtents()[1] == 3);
    assert(grid.getExtents()[2] == 4);
  }

  //
  // Accesors: Cartesian box
  //

  {
    const SizeList extents = {{2, 3, 4}};
    RegularGrid grid(extents,
                     BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});

    assert(grid.getDomain() ==
           (BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
  }


  //
  // Mathematical member functions
  //

  {
    const SizeList extents = {{2, 3, 4}};
    RegularGrid grid(extents, BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});
    const double eps = 10 * std::numeric_limits<double>::epsilon();

    //
    // Location to index
    //

    Point x = grid.getDomain().lower;
    grid.convertLocationToIndex(&x);
    assert(stlib::ext::euclideanDistance(x, Point{{0., 0., 0.}}) < eps);

    x = grid.getDomain().upper;
    grid.convertLocationToIndex(&x);
    Point upper;
    for (std::size_t i = 0; i != x.size(); ++i) {
      upper[i] = grid.getExtents()[i] - 1;
    }
    assert(stlib::ext::euclideanDistance(x, upper) < eps);

    //
    // Index to location
    //

    std::fill(x.begin(), x.end(), 0.);
    grid.convertIndexToLocation(&x);
    assert(stlib::ext::euclideanDistance(x, grid.getDomain().lower) < eps);

    x = upper;
    grid.convertIndexToLocation(&x);
    assert(stlib::ext::euclideanDistance(x, grid.getDomain().upper) < eps);

    //
    // convert()
    //

    grid.convert(IndexList{{0, 0, 0}}, &x);
    assert(stlib::ext::euclideanDistance(x, grid.getDomain().lower) < eps);

    grid.convert(ext::convert_array<Index>(grid.getExtents()) - Index(1), &x);
    assert(stlib::ext::euclideanDistance(x, grid.getDomain().upper) < eps);

    //
    // Vector to index
    //

    x = grid.getDomain().upper - grid.getDomain().lower;
    grid.convertVectorToIndex(&x);
    assert(stlib::ext::euclideanDistance(x, upper) < eps);

    //
    // computeRange().
    //

    Range range;

    grid.computeRange(BBox{{{-1., -1., -1.}}, {{0., 0., 0.}}}, &range);
    assert(range.bases() == (IndexList{{0, 0, 0}}));
    assert(range.extents() == (SizeList{{0, 0, 0}}));

    grid.computeRange(BBox{{{3., 4., 6.}}, {{4., 5., 7.}}}, &range);
    //assert(range.bases() == IndexList{{0, 0, 0}});
    assert(range.extents() == (SizeList{{0, 0, 0}}));

    grid.computeRange(BBox{{{1., 2., 3.}}, {{2.1, 3.1, 5.1}}}, &range);
    assert(range.bases() == (IndexList{{0, 0, 0}}));
    assert(range.extents() == grid.getExtents());

    grid.computeRange(BBox{{{1. - eps, 2. - eps, 3. - eps}},
        {{1. + eps, 2. + eps, 3. + eps}}}, &range);
    assert(range.bases() == (IndexList{{0, 0, 0}}));
    assert(range.extents() == (SizeList{{1, 1, 1}}));

    //
    // Range interface.
    //

    std::array<Point, 2> a = {{
        grid.getDomain().lower,
        grid.getDomain().upper
      }
    };
    std::array<Point, 2> b = a;
    grid.convertLocationsToIndices(b.begin(), b.end());
    grid.convertIndicesToLocations(b.begin(), b.end());
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(stlib::ext::euclideanDistance(a[i], b[i]) < eps);
    }
  }


  return 0;
}

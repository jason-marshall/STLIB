// -*- C++ -*-

#include "stlib/geom/grid/SimpleRegularGrid.h"
#include "stlib/geom/kernel/content.h"

#include "stlib/numerical/equality.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  using numerical::areEqual;

  typedef geom::SimpleRegularGrid<double, 3> SimpleRegularGrid;
  typedef SimpleRegularGrid::BBox BBox;
  typedef SimpleRegularGrid::IndexList IndexList;
  typedef SimpleRegularGrid::Range Range;
  typedef Range::Index Index;
  typedef BBox::Point Point;

  //
  // Constructors
  //

  {
    // Default Constructor.
    SimpleRegularGrid grid;
  }
  {
    // Construct from grid dimensions and Cartesian domain.
    const IndexList extents = {{2, 2, 2}};
    std::cout << "SimpleRegularGrid((2,2,2),BBox(0,0,0,1,1,1)) = " << '\n'
              << SimpleRegularGrid(extents,
                                   BBox{{{0., 0., 0.}}, {{1., 1., 1.}}})
              << '\n';
  }
  {
    // Copy constructor.
    const IndexList extents = {{2, 2, 2}};
    SimpleRegularGrid a(extents,
                        BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    SimpleRegularGrid b(a);
    assert(a == b);
  }
  {
    // Assignment operator.
    const IndexList extents = {{2, 2, 2}};
    SimpleRegularGrid a(extents,
                        BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    SimpleRegularGrid b = a;
    assert(a == b);
  }

  //
  // Accesors: grid size
  //

  {
    const IndexList extents = {{2, 3, 4}};
    SimpleRegularGrid grid(extents,
                           BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});

    assert(grid.getExtents()[0] == 2);
    assert(grid.getExtents()[1] == 3);
    assert(grid.getExtents()[2] == 4);
  }

  //
  // Accesors: Cartesian box
  //

  {
    const IndexList extents = {{2, 3, 4}};
    SimpleRegularGrid grid(extents,
                           BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});

    assert(grid.getLower() == (Point{{1., 2., 3.}}));
    assert(areEqual(grid.getDelta(),
                    Point{{1., 1., 2.}} /
                    ext::convert_array<double>(extents - Index(1))));
#if 0
    assert(grid.getDomain() ==
           (BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
#endif
  }


  //
  // Mathematical member functions
  //

  //
  // computeRange().
  //
  {
    // Unit extents, domain with zero content.
    const IndexList extents = {{1, 1, 1}};
    const BBox domain = {{{0., 0., 0.}}, {{0., 0., 0.}}};
    SimpleRegularGrid grid(extents, domain);
    Range range;

    range = grid.computeRange(BBox{{{1., 1., 1.}}, {{2., 2., 2.}}});
    assert(range.extents == (IndexList{{0, 0, 0}}));

    range = grid.computeRange(BBox{{{-1., -1., -1.}}, {{1., 1., 1.}}});
    assert(range.extents == (IndexList{{1, 1, 1}}));
    assert(range.bases == (IndexList{{0, 0, 0}}));
  }
  {
    // Unit extents, domain with zero content.
    const IndexList extents = {{1, 1, 1}};
    const BBox domain = {{{1., 1., 1.}}, {{1., 1., 1.}}};
    SimpleRegularGrid grid(extents, domain);
    Range range;

    range = grid.computeRange(BBox{{{2., 2., 2.}}, {{3., 3., 3.}}});
    assert(range.extents == (IndexList{{0, 0, 0}}));

    range = grid.computeRange(BBox{{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}});
    assert(range.extents == (IndexList{{1, 1, 1}}));
    assert(range.bases == (IndexList{{0, 0, 0}}));
  }
  {
    const IndexList extents = {{2, 3, 4}};
    const BBox domain = {{{1., 2., 3.}}, {{2., 3., 5.}}};
    SimpleRegularGrid grid(extents, domain);
    const double eps = 10 * std::numeric_limits<double>::epsilon();

    //
    // Location to index
    //

    Point x = domain.lower;
    grid.locationToIndex(&x);
    assert(stlib::ext::euclideanDistance(x, Point{{0., 0., 0.}}) < eps);

    x = domain.upper;
    grid.locationToIndex(&x);
    Point upper;
    for (std::size_t i = 0; i != x.size(); ++i) {
      upper[i] = grid.getExtents()[i] - 1;
    }
    assert(stlib::ext::euclideanDistance(x, upper) < eps);

    //
    // Index to location
    //

    std::fill(x.begin(), x.end(), 0.);
    grid.indexToLocation(&x);
    assert(stlib::ext::euclideanDistance(x, domain.lower) < eps);

    x = upper;
    grid.indexToLocation(&x);
    assert(stlib::ext::euclideanDistance(x, domain.upper) < eps);

    //
    // convert()
    //

    x = grid.indexToLocation(IndexList{{0, 0, 0}});
    assert(stlib::ext::euclideanDistance(x, domain.lower) < eps);

    x = grid.indexToLocation(ext::convert_array<Index>(grid.getExtents()) -
                             Index(1));
    assert(stlib::ext::euclideanDistance(x, domain.upper) < eps);

    //
    // Vector to index
    //

    x = domain.upper - domain.lower;
    grid.vectorToIndex(&x);
    assert(stlib::ext::euclideanDistance(x, upper) < eps);

    //
    // computeRange().
    //

    Range range;

    range = grid.computeRange(BBox{{{-1., -1., -1.}}, {{0., 0., 0.}}});
    assert(range.bases == (IndexList{{0, 0, 0}}));
    assert(range.extents == (IndexList{{0, 0, 0}}));

    range = grid.computeRange(BBox{{{3., 4., 6.}}, {{4., 5., 7.}}});
    //assert(range.bases == (IndexList{{0, 0, 0}}));
    assert(range.extents == (IndexList{{0, 0, 0}}));

    range = grid.computeRange(BBox{{{1., 2., 3.}}, {{2.1, 3.1, 5.1}}});
    assert(range.bases == (IndexList{{0, 0, 0}}));
    assert(range.extents == grid.getExtents());

    range = grid.computeRange(
      BBox{{{1. - eps, 2. - eps, 3. - eps}},
        {{1. + eps, 2. + eps, 3. + eps}}});
    assert(range.bases == (IndexList{{0, 0, 0}}));
    assert(range.extents == (IndexList{{1, 1, 1}}));

    //
    // Range interface.
    //

    std::array<Point, 2> a = {{domain.lower, domain.upper}};
    std::array<Point, 2> b = a;
    grid.locationsToIndices(b.begin(), b.end());
    grid.indicesToLocations(b.begin(), b.end());
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(stlib::ext::euclideanDistance(a[i], b[i]) < eps);
    }
  }


  return 0;
}

// -*- C++ -*-

#include "stlib/geom/kernel/distance.h"

#include "stlib/numerical/equality.h"

#include <iostream>
#include <limits>

#include <cmath>

using namespace stlib;

using namespace geom;
using numerical::areEqual;

int
main()
{
  // computeUpperBoundSquaredDistance for two boxes.
  {
    typedef geom::BBox<double, 3> BBox;
    // Adjust x coordinate.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
                     BBox{{{0., 0., 0.}}, {{1., 1., 1.}}}),
                    2.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
                     BBox{{{0.5, 0., 0.}}, {{1.5, 1., 1.}}}),
                    2.25));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
                     BBox{{{1., 0., 0.}}, {{2., 1., 1.}}}),
                    2.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
                     BBox{{{2., 0., 0.}}, {{3., 1., 1.}}}),
                    3.));

    // Adjust x and y coordinates.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                          {{1., 1., 1.}}},
                      BBox{{{1., 1., 0.}},
                          {{2., 2., 1.}}}),
                    5.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      BBox{{{1., 2., 0.}},
                          {{2., 3., 1.}}}),
                    6.));

    // Adjust all coordinates.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      BBox{{{1., 1., 1.}},
                          {{2., 2., 2.}}}),
                    8.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      BBox{{{2., 1., 1.}},
                          {{3., 2., 2.}}}),
                    9.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      BBox{{{2., 2., 1.}},
                          {{3., 3., 2.}}}),
                    14.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      BBox{{{2., 2., 2.}},
                          {{3., 3., 3.}}}),
                    19.));
  }
  // computeLowerBoundSquaredDistance for a box and a point.
  {
    typedef geom::BBox<double, 3> BBox;
    // Adjust x coordinate.
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                     {{0., 0., 0.}}),
                    0.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{0.5, 0., 0.}},
                        {{1.5, 1., 1.}}},
                     {{0., 0., 0.}}),
                    0.25 + 0. + 0.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{1., 0., 0.}},
                        {{2., 1., 1.}}},
                     {{0., 0., 0.}}),
                    1.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{2., 0., 0.}},
                        {{3., 1., 1.}}},
                     {{0., 0., 0.}}),
                    4.));

    // Adjust x and y coordinates.
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{1., 1., 0.}},
                        {{2., 2., 1.}}},
                     {{0., 0., 0.}}),
                    1. + 1. + 0.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{1., 2., 0.}},
                        {{2., 3., 1.}}},
                     {{0., 0., 0.}}),
                    1. + 4. + 0.));

    // Adjust all coordinates.
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{1., 1., 1.}},
                        {{2., 2., 2.}}},
                     {{0., 0., 0.}}),
                    1. + 1. + 1.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{2., 1., 1.}},
                        {{3., 2., 2.}}},
                     {{0., 0., 0.}}),
                    4. + 1. + 1.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{2., 2., 1.}},
                        {{3., 3., 2.}}},
                     {{0., 0., 0.}}),
                    4. + 4. + 1.));
    assert(areEqual(computeLowerBoundSquaredDistance
                    (BBox{{{2., 2., 2.}},
                        {{3., 3., 3.}}},
                     {{0., 0., 0.}}),
                    4. + 4. + 4.));
  }
  // computeUpperBoundSquaredDistance for a box and a point.
  {
    typedef geom::BBox<double, 3> BBox;
    typedef std::array<double, 3> Point;
    // Adjust x coordinate.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0., 0., 0.}},
                        {{1., 1., 1.}}},
                      Point{{0., 0., 0.}}),
                    2.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{0.5, 0., 0.}},
                        {{1.5, 1., 1.}}},
                      Point{{0., 0., 0.}}),
                    0.25 + 1. + 1.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{1., 0., 0.}},
                        {{2., 1., 1.}}},
                      Point{{0., 0., 0.}}),
                    3.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{2., 0., 0.}},
                        {{3., 1., 1.}}},
                      Point{{0., 0., 0.}}),
                    6.));

    // Adjust x and y coordinates.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{1., 1., 0.}},
                        {{2., 2., 1.}}},
                      Point{{0., 0., 0.}}),
                    1. + 1. + 4.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{1., 2., 0.}},
                        {{2., 3., 1.}}},
                      Point{{0., 0., 0.}}),
                    4. + 4. + 1.));

    // Adjust all coordinates.
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{1., 1., 1.}},
                        {{2., 2., 2.}}},
                      Point{{0., 0., 0.}}),
                    1. + 4. + 4.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{2., 1., 1.}},
                        {{3., 2., 2.}}},
                      Point{{0., 0., 0.}}),
                    4. + 4. + 4.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{2., 2., 1.}},
                        {{3., 3., 2.}}},
                      Point{{0., 0., 0.}}),
                    4. + 9. + 4.));
    assert(areEqual(computeUpperBoundSquaredDistance
                    (BBox{{{2., 2., 2.}},
                        {{3., 3., 3.}}},
                      Point{{0., 0., 0.}}),
                    4. + 9. + 9.));
  }
  //
  // 1-D
  //
  {
    typedef geom::BBox<double, 1> BBox;
    typedef std::array<double, 1> Point;

    assert(computeUpperBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{-1.}}) == 1);
    assert(computeUpperBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{4.}}) == 2);
    assert(computeUpperBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{0.}}) == 0);
    assert(computeUpperBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{1.}}) == 1);
    assert(computeUpperBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{2.}}) == 0);

    assert(computeLowerBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{-1.}}) == 1);
    assert(computeLowerBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{4.}}) == 2);
    assert(computeLowerBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{0.}}) == 0);
    assert(computeLowerBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{1.}}) == -1);
    assert(computeLowerBoundOnSignedDistance
           (BBox{{{0.}}, {{2.}}},
            Point{{2.}}) == 0);
  }

  //
  // 2-D
  //
  {
    typedef geom::BBox<double, 2> BBox;
    const double eps = 10 * std::numeric_limits<double>::epsilon();
    typedef std::array<double, 2> Point;

    //
    // Upper bound.
    // Outside the box.
    // Offset the point in one direction from a vertex of the box.
    //

    // Below, x.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 0.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 1.}}) -
                    std::sqrt(2.)) < eps);
    // Above, x.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 0.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 1.}}) -
                    std::sqrt(2.)) < eps);
    // Below, y.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., -1.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., -1.}}) -
                    std::sqrt(2.)) < eps);
    // Above, y.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 2.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 2.}}) -
                    std::sqrt(2.)) < eps);

    //
    // Upper bound.
    // Outside the box.
    // Offset the point outward from the midpoint of each side the box.
    //

    // Below, x.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 0.5}}) -
                    std::sqrt(1.25)) < eps);
    // Above, x.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 0.5}}) -
                    std::sqrt(1.25)) < eps);
    // Below, y.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, -1.}}) -
                    std::sqrt(1.25)) < eps);
    // Above, y.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, 2.}}) -
                    std::sqrt(1.25)) < eps);

    //
    // Upper bound.
    // Outside the box.
    // Offset the point diagonally outward from each vertex.
    //

    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., -1.}}) -
                    std::sqrt(5.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., -1.}}) -
                    std::sqrt(5.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 2.}}) -
                    std::sqrt(5.)) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 2.}}) -
                    std::sqrt(5.)) < eps);

    //
    // Upper bound.
    // Inside the box.
    //

    // Middle.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, 0.5}}) -
                    std::sqrt(0.5)) < eps);
    // Four corners.
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 0.}}) -
                    1) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 0.}}) -
                    1) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 1.}}) -
                    1) < eps);
    assert(std::abs(computeUpperBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 1.}}) -
                    1) < eps);


    //
    // Lower bound.
    // Outside the box.
    // Offset the point in one direction from a vertex of the box.
    //

    // Below, x.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 0.}}) -
                    1) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 1.}}) -
                    1) < eps);
    // Above, x.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 0.}}) -
                    1) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 1.}}) -
                    1) < eps);
    // Below, y.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., -1.}}) -
                    1) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., -1.}}) -
                    1) < eps);
    // Above, y.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 2.}}) -
                    1) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 2.}}) -
                    1) < eps);

    //
    // Upper bound.
    // Outside the box.
    // Offset the point outward from the midpoint of each side the box.
    //

    // Below, x.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 0.5}}) -
                    1) < eps);
    // Above, x.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 0.5}}) -
                    1) < eps);
    // Below, y.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, -1.}}) -
                    1) < eps);
    // Above, y.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, 2.}}) -
                    1) < eps);

    //
    // Upper bound.
    // Outside the box.
    // Offset the point diagonally outward from each vertex.
    //

    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., -1.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., -1.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{2., 2.}}) -
                    std::sqrt(2.)) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{-1., 2.}}) -
                    std::sqrt(2.)) < eps);

    //
    // Upper bound.
    // Inside the box.
    //

    // Middle.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0.5, 0.5}}) -
                    -0.5) < eps);
    // Four corners.
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 0.}}) -
                    0) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 0.}}) -
                    0) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{1., 1.}}) -
                    0) < eps);
    assert(std::abs(computeLowerBoundOnSignedDistance
                    (BBox{{{0., 0.}},
                        {{1., 1.}}},
                      Point{{0., 1.}}) -
                    0) < eps);
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/geom/mesh/iss/distinct_points.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  std::vector<std::array<double, 1> > points(10);
  const double minDistance = 0.5;
  std::vector<std::array<double, 1> > distinctPts;
  std::vector<std::size_t> indices;

  points[0][0] = 0;
  points[1][0] = 0.1;
  points[2][0] = 0.2;
  points[3][0] = 2;
  points[4][0] = 1.9;
  points[5][0] = 2.49;
  points[6][0] = 10;
  points[7][0] = 9.51;
  points[8][0] = 10.1;
  points[9][0] = 10.49;

  geom::buildDistinctPoints<1>(points.begin(), points.end(),
                               std::back_inserter(distinctPts),
                               std::back_inserter(indices), minDistance);

  assert(distinctPts.size() == 3);
  assert(indices.size() == 10);

  assert(distinctPts[0][0] == 0);
  assert(distinctPts[1][0] == 2);
  assert(distinctPts[2][0] == 10);

  assert(indices[0] == 0);
  assert(indices[1] == 0);
  assert(indices[2] == 0);
  assert(indices[3] == 1);
  assert(indices[4] == 1);
  assert(indices[5] == 1);
  assert(indices[6] == 2);
  assert(indices[7] == 2);
  assert(indices[8] == 2);
  assert(indices[9] == 2);

  distinctPts.clear();
  indices.clear();



  geom::buildDistinctPoints<1>(points.begin(), points.end(),
                               std::back_inserter(distinctPts),
                               std::back_inserter(indices));

  assert(distinctPts.size() == 10);
  assert(indices.size() == 10);

  for (std::size_t n = 0; n != 10; ++n) {
    assert(distinctPts[n] == points[n]);
    assert(indices[n] == n);
  }

  distinctPts.clear();
  indices.clear();




  const double eps = 10 * std::numeric_limits<double>::epsilon();
  points[0][0] = 0;
  points[1][0] = 0 + eps;
  points[2][0] = 0 - eps;
  points[3][0] = 2;
  points[4][0] = 2 - eps;
  points[5][0] = 2 - eps;
  points[6][0] = 10;
  points[7][0] = 10 + eps;
  points[8][0] = 10 + 2 * eps;
  points[9][0] = 10 - eps;

  geom::buildDistinctPoints<1>(points.begin(), points.end(),
                               std::back_inserter(distinctPts),
                               std::back_inserter(indices));

  assert(distinctPts.size() == 3);
  assert(indices.size() == 10);

  distinctPts.clear();
  indices.clear();

  return 0;
}

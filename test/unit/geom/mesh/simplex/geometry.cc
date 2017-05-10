// -*- C++ -*-

#include "stlib/geom/mesh/simplex/geometry.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  const double pi = 3.1415926535897932384626433832795;
  const double eps = 100 * std::numeric_limits<double>::epsilon();

  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    Tet tet = {{{{0., 0., 0.}},
               {{1., 0., 0.}},
               {{0., 1., 0.}},
               {{0., 0., 1.}}}};

    // Dihedral angle.
    assert(std::abs(geom::computeAngle(tet, 1, 2) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 1) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 3) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 2) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 1) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 1, 3) - pi / 2) < eps);

    //std::cout << geom::computeAngle(tet, 0, 1) << "\n";
    assert(std::abs(geom::computeAngle(tet, 0, 1) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 1, 0) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 0, 2) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 0) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 0, 3) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 0) -
                    std::atan(std::sqrt(2.0))) < eps);

    // Solid angle.
    assert(std::abs(geom::computeAngle(tet, 0) - pi / 2) < eps);
    const double x = 2 * std::atan(std::sqrt(2.0)) + pi / 2 - pi;
    assert(std::abs(geom::computeAngle(tet, 1) - x) < eps);
    assert(std::abs(geom::computeAngle(tet, 2) - x) < eps);
    assert(std::abs(geom::computeAngle(tet, 3) - x) < eps);
  }
  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    Tet tet = {{{{0., 0., 0.}},
               {{0., 1., 0.}},
               {{1., 0., 0.}},
               {{0., 0., 1.}}}};

    // Dihedral angle.
    assert(std::abs(geom::computeAngle(tet, 1, 2) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 1) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 3) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 2) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 1) - pi / 2) < eps);
    assert(std::abs(geom::computeAngle(tet, 1, 3) - pi / 2) < eps);

    //std::cout << geom::computeAngle(tet, 0, 1) << "\n";
    assert(std::abs(geom::computeAngle(tet, 0, 1) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 1, 0) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 0, 2) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 2, 0) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 0, 3) -
                    std::atan(std::sqrt(2.0))) < eps);
    assert(std::abs(geom::computeAngle(tet, 3, 0) -
                    std::atan(std::sqrt(2.0))) < eps);

    // Solid angle.
    assert(std::abs(geom::computeAngle(tet, 0) - pi / 2) < eps);
    const double x = 2 * std::atan(std::sqrt(2.0)) + pi / 2 - pi;
    assert(std::abs(geom::computeAngle(tet, 1) - x) < eps);
    assert(std::abs(geom::computeAngle(tet, 2) - x) < eps);
    assert(std::abs(geom::computeAngle(tet, 3) - x) < eps);
  }
  // 2-D.
  {
    const std::size_t N = 2;
    typedef std::array<double, N> Point;
    typedef std::array < Point, N + 1 > Simplex;

    Simplex s = {{{{0., 0.}},
                 {{1., 0.}},
                 {{0., 1.}}}};

    // Angle.
    assert(std::abs(geom::computeAngle(s, 0) - 0.5 * pi) < eps);
    assert(std::abs(geom::computeAngle(s, 1) - 0.25 * pi) < eps);
    assert(std::abs(geom::computeAngle(s, 2) - 0.25 * pi) < eps);
  }
  // 1-D.
  {
    const std::size_t N = 1;
    typedef std::array<double, N> Point;
    typedef std::array < Point, N + 1 > Simplex;

    Simplex s = {{{{0.}},
                 {{1.}}}};

    // Angle.
    assert(geom::computeAngle(s, 0) == 1);
    assert(geom::computeAngle(s, 1) == 1);
  }

  return 0;
}

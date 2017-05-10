// -*- C++ -*-

#include "stlib/geom/mesh/simplex/functor.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // Determinant.
  //
  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    const double eps = 100.0 * std::numeric_limits<double>::epsilon();

    Tet tet = {{{{0., 0., 0.}},
               {{1., 0., 0.}},
               {{0., 1., 0.}},
               {{0., 0., 1.}}}};

    {
      geom::SimplexDeterminant<3> f;
      assert(std::abs(f(tet) - std::sqrt(2.0)) < eps);
      geom::SimplexDeterminant<3> g(f);
      assert(std::abs(g(tet) - std::sqrt(2.0)) < eps);
    }
    {
      geom::SimplexDeterminant<3> f = geom::simplexDeterminant<3, 3, double>();
      assert(std::abs(f(tet) - std::sqrt(2.0)) < eps);
    }
  }

  //
  // Content.
  //
  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    const double eps = 100.0 * std::numeric_limits<double>::epsilon();

    Tet tet = {{{{0., 0., 0.}},
               {{1., 0., 0.}},
               {{0., 1., 0.}},
               {{0., 0., 1.}}}};

    {
      geom::SimplexContent<3> f;
      assert(std::abs(f(tet) - 1.0 / 6.0) < eps);
      geom::SimplexContent<3> g(f);
      assert(std::abs(g(tet) - 1.0 / 6.0) < eps);
    }
    {
      geom::SimplexContent<3> f = geom::simplexContent<3, 3, double>();
      assert(std::abs(f(tet) - 1.0 / 6.0) < eps);
    }
  }

  //
  // Minimum edge length.
  //
  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    const double eps = 10.0 * std::numeric_limits<double>::epsilon();

    Tet tet = {{{{0., 0., 0.}},
               {{1., 0., 0.}},
               {{0., 1., 0.}},
               {{0., 0., 1.}}}};

    {
      geom::SimplexMinimumEdgeLength<3> f;
      assert(std::abs(f(tet) - 1.0) < eps);
      geom::SimplexMinimumEdgeLength<3> g(f);
      assert(std::abs(g(tet) - 1.0) < eps);
    }
    {
      geom::SimplexMinimumEdgeLength<3> f
        = geom::simplexMinimumEdgeLength<3, 3, double>();
      assert(std::abs(f(tet) - 1.0) < eps);
    }
  }

  //
  // Maximum edge length.
  //
  {
    typedef std::array<double, 3> Point;
    typedef std::array < Point, 3 + 1 > Tet;

    const double eps = 10.0 * std::numeric_limits<double>::epsilon();

    Tet tet = {{{{0., 0., 0.}},
               {{1., 0., 0.}},
               {{0., 1., 0.}},
               {{0., 0., 1.}}}};

    {
      geom::SimplexMaximumEdgeLength<3> f;
      assert(std::abs(f(tet) - std::sqrt(2.0)) < eps);
      geom::SimplexMaximumEdgeLength<3> g(f);
      assert(std::abs(g(tet) - std::sqrt(2.0)) < eps);
    }
    {
      geom::SimplexMaximumEdgeLength<3> f
        = geom::simplexMaximumEdgeLength<3, 3, double>();
      assert(std::abs(f(tet) - std::sqrt(2.0)) < eps);
    }
  }

  return 0;
}

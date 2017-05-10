// -*- C++ -*-

#include "stlib/geom/kernel/BallSquared.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{
  typedef geom::BallSquared<double, 3> BallSquared;
  typedef BallSquared::Point Point;
  {
    // Default constructor
    BallSquared x;
    std::cout << "BallSquared() = " << x << "\n";
  }
  {
    // Point constructor
    const Point c = {{1, 2, 3}};
    const double r2 = 1;
    const BallSquared x = {c, r2};
    std::cout << "BallSquared((1,2,3),1) = " << x << "\n";

    // copy constructor
    const BallSquared y(x);
    assert(y == x);
    std::cout << "copy = " << y << "\n";

    // assignment operator
    const BallSquared z = x;
    assert(z == x);
    std::cout << "assignment = " << z << "\n";

    // Accessors.
    assert(x.center == c);
    assert(x.squaredRadius == r2);
    std::cout << "Accessors: " << "\n"
              << "center = " << x.center << "\n"
              << "radius = " << x.squaredRadius << "\n";
  }
  // == operator
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{2., 3., 5.}}, 1};
    assert(!(a == b));
  }
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{1., 2., 3.}}, 2};
    assert(!(a == b));
  }
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{1., 2., 3.}}, 1};
    assert(a == b);
  }
  // != operator
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{2., 3., 5.}}, 1};
    assert(a != b);
  }
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{1., 2., 3.}}, 2};
    assert(a != b);
  }
  {
    BallSquared a = {{{1., 2., 3.}}, 1};
    BallSquared b = {{{1., 2., 3.}}, 1};
    assert(!(a != b));
  }

  return 0;
}

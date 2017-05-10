// -*- C++ -*-

#include "stlib/mst/Atom.h"

using namespace stlib;

int
main()
{
  typedef double Number;
  typedef geom::Ball<Number, 3> Atom;
  typedef Atom::Point Point;

  {
    // Construct from the center and radius.
    Atom x = {{{1, 2, 3}}, 4};
    assert(x.center == (Point{{1, 2, 3}}));
    assert(x.radius == 4);
    {
      // Copy constructor.
      Atom y(x);
      assert(y == x);
    }
    {
      // Assignment operator.
      Atom y;
      y = x;
      assert(y == x);
    }
  }

  {
    // isInside
    const Atom x = {{{0, 0, 0}}, 1};
    assert(isInside(x, Point{{0, 0, 0}}));
    assert(isInside(x, Point{{0.99, 0, 0}}));
    assert(isInside(x, Point{{0, 0.99, 0}}));
    assert(isInside(x, Point{{0, 0, 0.99}}));
    assert(! isInside(x, Point{{1.01, 0, 0}}));
    assert(! isInside(x, Point{{0, 1.01, 0}}));
    assert(! isInside(x, Point{{0, 0, 1.01}}));
  }

  // doesClip
  {
    // Disjoint.
    const Atom x = {{{0, 0, 0}}, 1}, y = {{{2, 0, 0}}, 0.99};
    assert(! doesClip(x, y));
  }
  {
    // Touching.
    const Atom x = {{{0, 0, 0}}, 1}, y = {{{2, 0, 0}}, 1};
    assert(! doesClip(x, y));
  }
  {
    // Intersecting.
    const Atom x = {{{0, 0, 0}}, 1}, y = {{{2, 0, 0}}, 1.01};
    assert(doesClip(x, y));
  }
  {
    // Intersecting.
    const Atom x = {{{0, 0, 0}}, 1.01}, y = {{{2, 0, 0}}, 1};
    assert(doesClip(x, y));
  }
  {
    // x inside y.
    const Atom x = {{{0, 0, 0}}, 1}, y = {{{0, 0, 0}}, 2};
    assert(doesClip(x, y));
  }
  {
    // y inside x.
    const Atom x = {{{0, 0, 0}}, 2}, y = {{{0, 0, 0}}, 1};
    assert(! doesClip(x, y));
  }

  return 0;
}

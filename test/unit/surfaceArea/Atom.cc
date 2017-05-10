// -*- C++ -*-

#include "stlib/surfaceArea/Atom.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef float Number;
  typedef surfaceArea::LogicalAtom LogicalAtom;
  typedef surfaceArea::PhysicalAtom<Number> PhysicalAtom;
  // A Cartesian point.
  typedef std::array<Number, 3> Point;

  {
    const std::size_t atomIndex = 7;
    const std::size_t polarIndex = 0;
    const Point center = {{2, 3, 5}};
    {
      LogicalAtom x = {atomIndex, polarIndex};
      assert(x.atomIndex == atomIndex);
      assert(x.polarIndex == polarIndex);
      PhysicalAtom y = x;
      y.center = center;
      assert(y.atomIndex == atomIndex);
      assert(y.polarIndex == polarIndex);
      assert(y.center == center);
      PhysicalAtom z;
      z = x;
      z.center = center;
      assert(z.atomIndex == atomIndex);
      assert(z.polarIndex == polarIndex);
      assert(z.center == center);
    }
  }

  return 0;
}

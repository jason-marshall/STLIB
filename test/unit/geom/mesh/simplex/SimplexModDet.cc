// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexModDet.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using namespace geom;

  for (double d = 1; d > std::numeric_limits<double>::epsilon(); d *= 0.1) {
    std::cout << "d = " << d
              << ", delta = " << SimplexModDet<>::getDelta(d)
              << ", h = " << SimplexModDet<>::getH(d, d)
              << '\n';
  }
  for (double d = 1; d > std::numeric_limits<double>::epsilon(); d *= 0.1) {
    std::cout << "d = " << -d
              << ", delta = " << SimplexModDet<>::getDelta(-d)
              << ", h = " << SimplexModDet<>::getH(-d, -d)
              << '\n';
  }
  return 0;
}

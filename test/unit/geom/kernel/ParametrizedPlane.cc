// -*- C++ -*-

#include "stlib/geom/kernel/ParametrizedPlane.h"

#include <iostream>

using namespace stlib;
using namespace geom;

int
main()
{
  //
  // 3-D
  //
  {
    typedef ParametrizedPlane<3, double> Plane;
    typedef Plane::Point Point;
    typedef Plane::ParameterPoint ParameterPoint;
    {
      // Default constructor
      Plane x;
    }
    {
      // Point constructor
      Point origin = {{1, 2, 3}}, axis0 = {{2, 0, 0}}, axis1 = {{0, 3, 0}};
      Plane x(origin, origin + axis0, origin + axis1);
      // copy constructor
      Plane copy(x);
      assert(copy == x);
      // assignment operator
      Plane assign;
      assign = x;
      assert(assign == x);

      // accessors
      assert(stlib::ext::euclideanDistance(x.computePosition
                               (ParameterPoint{{0., 0.}}),
                               origin) < 1e-8);
      assert(stlib::ext::euclideanDistance(x.computePosition
                               (ParameterPoint{{1., 0.}}),
                               origin + axis0) < 1e-8);
      assert(stlib::ext::euclideanDistance(x.computePosition
                               (ParameterPoint{{0., 1.}}),
                               origin + axis1) < 1e-8);
      Point d0, d1;
      x.computeDerivative(ParameterPoint{{0., 0.}}, &d0, &d1);
      assert(stlib::ext::euclideanDistance(d0, axis0) < 1e-8);
      assert(stlib::ext::euclideanDistance(d1, axis1) < 1e-8);
    }
  }

  return 0;
}

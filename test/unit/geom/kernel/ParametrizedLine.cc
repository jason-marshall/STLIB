// -*- C++ -*-

#include "stlib/geom/kernel/ParametrizedLine.h"
#include "stlib/geom/kernel/content.h"

#include <iostream>

using namespace stlib;
using namespace geom;

int
main()
{
  //
  // 2-D
  //
  {
    typedef ParametrizedLine<2, double> Line;
    typedef Line::Point Point;
    typedef Line::ParameterPoint ParameterPoint;
    {
      // Default constructor
      Line x;
    }
    {
      // Point constructor
      Point source = {{1, 2}}, target = {{1, 3}}, tangent = {{0, 1}};
      Line ln(source, target);
      // copy constructor
      Line copy(ln);
      assert(copy == ln);
      // assignment operator
      Line assign;
      assign = ln;
      assert(assign == ln);

      // accessors
      assert(stlib::ext::euclideanDistance(ln.computePosition(ParameterPoint{{0.}}),
                               source) < 1e-8);
      assert(stlib::ext::euclideanDistance(ln.computePosition(ParameterPoint{{1.}}),
                               target) < 1e-8);
      assert(stlib::ext::euclideanDistance(ln.computeDerivative(ParameterPoint{{0.}}),
                               tangent) < 1e-8);
    }
  }


  //
  // 3-D
  //
  {
    typedef ParametrizedLine<3, double> Line;
    typedef Line::Point Point;
    typedef Line::ParameterPoint ParameterPoint;
    {
      // Default constructor
      Line x;
    }
    {
      // Point constructor
      Point source = {{1, 2, 3}}, target = {{2, 3, 4}}, tangent = {{1, 1, 1}};
      tangent /= std::sqrt(3.);
      Line ln(source, target);
      // copy constructor
      Line copy(ln);
      assert(copy == ln);
      // assignment operator
      Line assign;
      assign = ln;
      assert(assign == ln);

      // accessors
      assert(stlib::ext::euclideanDistance(ln.computePosition(ParameterPoint{{0.}}),
                               source) < 1e-8);
      assert(stlib::ext::euclideanDistance(ln.computePosition(ParameterPoint{{1.}}),
                               target) < 1e-8);
      assert(stlib::ext::euclideanDistance(ln.computeDerivative(ParameterPoint{{0.}}),
                               Point{{1., 1., 1.}}) < 1e-8);
    }
  }

  return 0;
}

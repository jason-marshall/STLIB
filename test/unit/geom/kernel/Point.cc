// -*- C++ -*-

#include "stlib/geom/kernel/Point.h"

#include "stlib/numerical/constants.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  const double eps = 10 * std::numeric_limits<double>::epsilon();

  //
  // Math operators.
  //

  // Compute an orthogonal vector.
  {
    std::array<double, 3> x = {{1, 0, 0}};
    std::array<double, 3> y = {{0, 0, 0}};
    geom::computeAnOrthogonalVector(x, &y);
    assert(x[0] == 1 && x[1] == 0 && x[2] == 0);
    assert(y[0] == 0 && y[1] == 1 && y[2] == 0);
    assert(std::abs(stlib::ext::dot(x, y)) < eps);
  }
  {
    std::array<double, 3> x = {{ -1, 0, 0}};
    std::array<double, 3> y = {{0, 0, 0}};
    geom::computeAnOrthogonalVector(x, &y);
    assert(std::abs(stlib::ext::dot(x, y)) < eps);
  }
  {
    std::array<double, 3> x = {{0, 1, 0}}, y;
    geom::computeAnOrthogonalVector(x, &y);
    stlib::ext::normalize(&y);
    assert(std::abs(stlib::ext::dot(x, y)) < eps);
  }
  {
    std::array<double, 3> x = {{0, 0, 1}}, y;
    geom::computeAnOrthogonalVector(x, &y);
    stlib::ext::normalize(&y);
    assert(std::abs(stlib::ext::dot(x, y)) < eps);
  }

  // squared distance
  {
    std::array<double, 1> a = {{1}};
    std::array<double, 1> b = {{2}};
    assert(std::abs(stlib::ext::squaredDistance(a, b) - 1) < 1 * eps);
  }
  {
    std::array<double, 2> a = {{1, 2}};
    std::array<double, 2> b = {{2, 3}};
    assert(std::abs(stlib::ext::squaredDistance(a, b) - 2) < 2 * eps);
  }
  {
    std::array<double, 3> a = {{1, 2, 3}};
    std::array<double, 3> b = {{2, 3, 5}};
    assert(std::abs(stlib::ext::squaredDistance(a, b) - 6) < 6 * eps);
  }
  {
    std::array<double, 4> a = {{1, 2, 3, 4}};
    std::array<double, 4> b = {{2, 3, 5, 7}};
    assert(std::abs(stlib::ext::squaredDistance(a, b) - 15) < 15 * eps);
  }

  // sign of turn
  {
    typedef std::array<double, 2> Point;
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{1., 1.}};
      assert((geom::computeSignOfTurn(a, b, c) == -1));
    }
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{2., 0.}};
      assert(geom::computeSignOfTurn(a, b, c) == 0);
    }
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{1., -1.}};
      assert(geom::computeSignOfTurn(a, b, c) == 1);
    }
  }
  {
    // approximate sign of turn
    typedef std::array<double, 2> Point;
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{1., 1.}};
      assert(geom::computeApproximateSignOfTurn(a, b, c) == -1);
    }
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{2., eps}};
      assert(geom::computeApproximateSignOfTurn(a, b, c) == 0);
    }
    {
      Point a = {{0., 0.}};
      Point b = {{1., 0.}};
      Point c = {{1., -1.}};
      assert(geom::computeApproximateSignOfTurn(a, b, c) == 1);
    }
  }
  {
    // sqared_magnitude()
    std::array<double, 3> a = {{1, 2, 3}};
    assert(stlib::ext::squaredMagnitude(a) == 14);
  }
  {
    // magnitude()
    std::array<double, 3> a = {{1, 2, 3}};
    assert(std::abs(stlib::ext::magnitude(a) - std::sqrt(14.)) <
           std::sqrt(14.) * eps);
  }
  {
    // normalize()
    std::array<double, 3> a = {{1, 2, 3}};
    stlib::ext::normalize(&a);
    assert(std::abs(stlib::ext::magnitude(a) - 1) < 1 * eps);
  }
  {
    // rotate pi/2
    std::array<double, 2> a = {{1, 2}};
    geom::rotatePiOver2(&a);
    std::array<double, 2> b = {{ -2, 1}};
    assert(a == b);
  }
  {
    // rotate -pi/2
    std::array<double, 2> a = {{1, 2}};
    geom::rotateMinusPiOver2(&a);
    assert((a == std::array<double, 2>{{2., -1.}}));
  }
  {
    // dot
    std::array<double, 3> a = {{1, 2, 3}};
    std::array<double, 3> b = {{2, 3, 5}};
    assert(stlib::ext::dot(a, b) == 23);
  }
  {
    // cross
    std::array<double, 3> a = {{1, 2, 3}};
    std::array<double, 3> b = {{2, 3, 5}};
    assert((stlib::ext::cross(a, b) == (std::array<double, 3>{{1., 1., -1.}})));
  }
  {
    // Geometry::triple_product
    std::array<double, 3> a = {{1, 2, 3}};
    std::array<double, 3> b = {{2, 3, 5}};
    std::array<double, 3> c = {{1, 3, 5}};
    assert(stlib::ext::tripleProduct(a, b, c) == -1);
  }
  {
    // discriminant
    std::array<double, 2> a = {{1, 2}};
    std::array<double, 2> b = {{3, 4}};
    assert(stlib::ext::discriminant(a, b) == -2);
  }
  {
    // pseudo angle
    typedef std::array<double, 2> Point;

    assert(geom::computePseudoAngle(Point{{0., 0.}}) == 0);
    assert(geom::computePseudoAngle(Point{{0., 1.}}) == 1);
    assert(geom::computePseudoAngle(Point{{-1., 0.}}) == 2);
    assert(geom::computePseudoAngle(Point{{0., -1.}}) == 3);
  }
  {
    // angle
    // 2-D
    {
      std::array<double, 2> a = {{1, 0}};
      std::array<double, 2> b = {{1, 0}};
      assert(std::abs(geom::computeAngle(a, b) - 0) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 2> a = {{2, 0}};
      std::array<double, 2> b = {{1, 0}};
      assert(std::abs(geom::computeAngle(a, b) - 0) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 2> a = {{1, 0}};
      std::array<double, 2> b = {{2, 0}};
      assert(std::abs(geom::computeAngle(a, b) - 0) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 2> a = {{1, 0}};
      std::array<double, 2> b = {{0, 1}};
      assert(std::abs(geom::computeAngle(a, b) -
                      numerical::Constants<double>::Pi() / 2) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 2> a = {{1, 0}};
      std::array<double, 2> b = {{ -1, 0}};
      assert(std::abs(geom::computeAngle(a, b) -
                      numerical::Constants<double>::Pi()) <
             10 * std::numeric_limits<double>::epsilon());
    }
    // 3-D
    {
      std::array<double, 3> a = {{1, 0, 0}};
      std::array<double, 3> b = {{1, 0, 0}};
      assert(std::abs(geom::computeAngle(a, b) - 0) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 3> a = {{1, 0, 0}};
      std::array<double, 3> b = {{0, 1, 0}};
      assert(std::abs(geom::computeAngle(a, b) -
                      numerical::Constants<double>::Pi() / 2) <
             10 * std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 3> a = {{1, 0, 0}};
      std::array<double, 3> b = {{0, 0, 1}};
      assert(std::abs(geom::computeAngle(a, b) -
                      numerical::Constants<double>::Pi() / 2) <
             10 * std::numeric_limits<double>::epsilon());
    }
  }

  return 0;
}

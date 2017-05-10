// -*- C++ -*-

#include "stlib/geom/kernel/Ball.h"

#include "stlib/numerical/equality.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

using numerical::areEqual;

int
main()
{
  typedef float Number;
  typedef geom::Ball<Number, 3> Ball;
  typedef Ball::Point Point;
  {
    // Default constructor
    Ball x;
    std::cout << "Ball() = " << x << "\n";
  }
  {
    // Initializer list.
    const Point c = {{1, 2, 3}};
    const Number r = 1;
    const Ball x = {c, r};
    std::cout << "Ball((1,2,3),1) = " << x << "\n";

    // copy constructor
    const Ball y(x);
    assert(y == x);
    std::cout << "copy = " << y << "\n";

    // assignment operator
    const Ball z = x;
    assert(z == x);
    std::cout << "assignment = " << z << "\n";

    // Accessors.
    assert(x.center == c);
    assert(x.radius == r);
    std::cout << "Accessors: " << "\n"
              << "center = " << x.center << "\n"
              << "radius = " << x.radius << "\n";
  }
  // == operator
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{2., 3., 5.}}, 1};
    assert(!(a == b));
  }
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{1., 2., 3.}}, 2};
    assert(!(a == b));
  }
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{1., 2., 3.}}, 1};
    assert(a == b);
  }
  // != operator
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{2., 3., 5.}}, 1};
    assert(a != b);
  }
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{1., 2., 3.}}, 2};
    assert(a != b);
  }
  {
    Ball a = {{{1., 2., 3.}}, 1};
    Ball b = {{{1., 2., 3.}}, 1};
    assert(!(a != b));
  }
  // bbox()
  {
    Ball ball = {{{1., 2., 3.}}, 1};
    geom::BBox<Number, 3> box =
      geom::specificBBox<geom::BBox<Number, 3> >(ball);
    assert(box.lower == (Point{{0., 1., 2.}}));
    assert(box.upper == (Point{{2., 3., 4.}}));
  }
  {
    std::vector<Ball> balls;
    balls.push_back(Ball{{{1., 2., 3.}}, Number(1)});
    {
      geom::BBox<Number, 3> box =
        geom::specificBBox<geom::BBox<Number, 3> >(balls.begin(), balls.end());
      assert(box.lower == (Point{{0., 1., 2.}}));
      assert(box.upper == (Point{{2., 3., 4.}}));
    }
    balls.push_back(Ball{{{2., 3., 5.}}, Number(1)});
    {
      geom::BBox<Number, 3> box =
        geom::specificBBox<geom::BBox<Number, 3> >(balls.begin(), balls.end());
      assert(box.lower == (Point{{0., 1., 2.}}));
      assert(box.upper == (Point{{3., 4., 6.}}));
    }
  }
  // bbox()
  {
    Ball ball = {{{1., 2., 3.}}, 1};
    geom::BBox<Number, 3> const box =
      geom::specificBBox<geom::BBox<Number, 3> >(ball);
    assert(box.lower == (Point{{0., 1., 2.}}));
    assert(box.upper == (Point{{2., 3., 4.}}));
  }
  // doIntersect()
  {
    Ball a = {{{0., 0., 0.}}, 1};
    Ball b = {{{1., 0., 0.}}, 1};
    assert(doIntersect(a, b));
  }
  {
    Ball a = {{{0., 0., 0.}}, 1};
    Ball b = {{{3., 0., 0.}}, 1};
    assert(! doIntersect(a, b));
  }

  // closestPoint()
  {
    Ball ball = {{{0., 0., 0.}}, 2};
    Point x = {{0, 0, 0}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(-2)));
    assert(areEqual(closest, Point{{2., 0., 0.}}));
  }
  {
    Ball ball = {{{0., 0., 0.}}, 2};
    Point x = {{0.5, 0, 0}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(-1.5)));
    assert(areEqual(closest, Point{{2., 0., 0.}}));
  }
  {
    Ball ball = {{{0., 0., 0.}}, 2};
    Point x = {{4, 0, 0}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(2)));
    assert(areEqual(closest, Point{{2., 0., 0.}}));
  }
  {
    Ball ball = {{{1., 2., 3.}}, 2};
    Point x = {{1, 2, 3}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(-2)));
    assert(areEqual(closest, Point{{3., 2., 3.}}));
  }
  {
    Ball ball = {{{1., 2., 3.}}, 2};
    Point x = {{1.5, 2, 3}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(-1.5)));
    assert(areEqual(closest, Point{{3., 2., 3.}}));
  }
  {
    Ball ball = {{{1., 2., 3.}}, 2};
    Point x = {{5, 2, 3}};
    Point closest;
    assert(areEqual(closestPoint(ball, x, &closest), Number(2)));
    assert(areEqual(closest, Point{{3., 2., 3.}}));
  }

  return 0;
}

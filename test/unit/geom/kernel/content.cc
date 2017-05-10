// -*- C++ -*-

#include "stlib/geom/kernel/content.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  //
  // Distance
  //
  {
    // 1-D Point, 1-simplex.
    typedef std::array<double, 1> Point;
    typedef std::array<Point, 2> Simplex;
    Point x = {{0}}, y = {{1}};
    assert(geom::computeContent(x, y) == 1);
    assert(geom::computeContent(y, x) == -1);
    Simplex s = {{x, y}};
    assert(geom::computeContent(s) == 1);

    Point gradient;
    geom::computeGradientOfContent(x, y, &gradient);
    assert(gradient == Point{{-1.}});
    geom::computeGradientOfContent(s, &gradient);
    assert(gradient == Point{{-1.}});
  }
  {
    // 2-D Point, 1-simplex.
    typedef std::array<double, 2> Point;
    typedef std::array<Point, 2> Simplex;
    Point x = {{0, 0}}, y = {{1, 0}};
    assert(geom::computeContent(x, y) == 1);
    assert(geom::computeContent(y, x) == 1);
    Simplex s = {{x, y}};
    assert(geom::computeContent(s) == 1);

    Point gradient;
    geom::computeGradientOfContent(x, y, &gradient);
    assert(stlib::ext::euclideanDistance(gradient, Point{{-1., 0.}}) <
           10 * std::numeric_limits<double>::epsilon());
  }

  //
  // Area
  //
  {
    // 2-D Point, 2-simplex.
    typedef std::array<double, 2> Point;
    typedef std::array<Point, 3> Simplex;
    Point x = {{0, 0}}, y = {{1, 0}}, z = {{0, 1}};
    assert(std::abs(geom::computeContent(x, y, z) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(y, z, x) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(z, x, y) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(x, z, y) + 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(z, y, x) + 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(y, x, z) + 0.5) < 1e-8);
    {
      Simplex s = {{x, y, z}};
      assert(std::abs(geom::computeContent(s) - 0.5) < 1e-8);
      assert(std::abs(geom::computeSquaredContent(s) - 0.25) < 1e-8);
    }
    {
      Simplex s = {{y, x, z}};
      assert(std::abs(geom::computeContent(s) + 0.5) < 1e-8);
      assert(std::abs(geom::computeSquaredContent(s) - 0.25) < 1e-8);
    }

    Point gradient;
    geom::computeGradientOfContent(x, y, z, &gradient);
    assert(stlib::ext::euclideanDistance(gradient, Point{{-0.5, -0.5}}) <
           10 * std::numeric_limits<double>::epsilon());
  }
  {
    // 3-D Point, 2-simplex.
    typedef std::array<double, 3> Point;
    typedef std::array<Point, 3> Simplex;
    Point x = {{0, 0, 0}}, y = {{1, 0, 0}}, z = {{0, 1, 0}};
    assert(std::abs(geom::computeContent(x, y, z) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(y, z, x) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(z, x, y) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(x, z, y) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(z, y, x) - 0.5) < 1e-8);
    assert(std::abs(geom::computeContent(y, x, z) - 0.5) < 1e-8);
    {
      Simplex s = {{x, y, z}};
      assert(std::abs(geom::computeContent(s) - 0.5) < 1e-8);
      assert(std::abs(geom::computeSquaredContent(s) - 0.25) < 1e-8);
    }
    {
      Simplex s = {{y, x, z}};
      assert(std::abs(geom::computeContent(s) - 0.5) < 1e-8);
      assert(std::abs(geom::computeSquaredContent(s) - 0.25) < 1e-8);
    }

    Point gradient;
    geom::computeGradientOfContent(x, y, z, &gradient);
    assert(stlib::ext::euclideanDistance(gradient, Point{{-0.5, -0.5, 0.}}) <
           10 * std::numeric_limits<double>::epsilon());
  }

  //
  // Volume
  //
  {
    // 3-D Point, 3-simplex.
    typedef std::array<double, 3> Point;
    typedef std::array<Point, 4> Simplex;
    Point a = {{0, 0, 0}}, b = {{1, 0, 0}}, c = {{0, 1, 0}}, d = {{0, 0, 1}};
    assert(std::abs(geom::computeContent(a, b, c, d) - 1. / 6) < 1e-8);
    assert(std::abs(geom::computeContent(b, a, c, d) + 1. / 6) < 1e-8);
    {
      Simplex s = {{a, b, c, d}};
      assert(std::abs(geom::computeContent(s) - 1. / 6) < 1e-8);
    }
    {
      Simplex s = {{b, a, c, d}};
      assert(std::abs(geom::computeContent(s) + 1. / 6) < 1e-8);
    }

    Point gradient;
    geom::computeGradientOfContent(a, b, c, d, &gradient);
    assert(stlib::ext::euclideanDistance(gradient, Point{{-1. / 6., -1. / 6.,
              -1. / 6.}}) <
           10 * std::numeric_limits<double>::epsilon());
  }
  {
    // 3-D Point, 3-simplex.
    typedef std::array<double, 3> Point;
    Point a = {{1, 2, 3}}, b = {{2, 2, 3}}, c = {{1, 3, 3}}, d = {{1, 2, 4}};
    assert(std::abs(geom::computeContent(a, b, c, d) - 1. / 6) < 1e-8);
    assert(std::abs(geom::computeContent(b, a, c, d) + 1. / 6) < 1e-8);
  }
  // CONTINUE Added tests for unsigned volume.
  return 0;
}

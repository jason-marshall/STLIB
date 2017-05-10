// -*- C++ -*-

#include "stlib/numerical/interpolation/QuinticInterpolation2D.h"
#include "stlib/numerical/derivative/ridders.h"
#include "stlib/numerical/equality.h"
#include "stlib/numerical/constants.h"

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _T, bool _IsPeriodic = false>
class QuinticInterpolation2DTest :
  public numerical::QuinticInterpolation2D<_T, _IsPeriodic>
{
  //
  // Types.
  //
private:

  typedef numerical::QuinticInterpolation2D<_T, _IsPeriodic> Base;

public:

  typedef typename Base::Value Value;
  typedef typename Base::ValueGradientHessian ValueGradientHessian;
  typedef typename Base::Point Point;
  typedef typename Base::BBox BBox;
  typedef typename Base::ValueGrid ValueGrid;
  typedef typename Base::SizeList SizeList;
  typedef typename Base::Dt2d3 Dt2d3;

  //
  // Data.
  //
public:

  using Base::_grid;
  using Base::_inverseWidths;

  //--------------------------------------------------------------------------
  // Constructors etc.
public:

  QuinticInterpolation2DTest() :
    Base()
  {
  }

  QuinticInterpolation2DTest(const ValueGrid& valueGrid, const BBox& domain) :
    Base(valueGrid, domain)
  {
  }

  using Base::interpolate;
};


template<typename _T>
bool
areEqual(const numerical::ValueGradientHessian<_T>& a,
         const numerical::ValueGradientHessian<_T>& b,
         const _T toleranceFactor = 1000)
{
  using numerical::areEqual;

  return areEqual(a.f, b.f, toleranceFactor) &&
         areEqual(a.fx, b.fx, toleranceFactor) &&
         areEqual(a.fy, b.fy, toleranceFactor) &&
         areEqual(a.fxx, b.fxx, toleranceFactor) &&
         areEqual(a.fxy, b.fxy, toleranceFactor) &&
         areEqual(a.fyy, b.fyy, toleranceFactor);
}


template<typename _Interpolant, std::size_t _Order>
class InterpolantValue :
  public std::unary_function<std::array<double, 2>, double>
{

  _Interpolant _interpolant;

public:

  InterpolantValue(const _Interpolant& interpolant) :
    _interpolant(interpolant)
  {
  }

  double
  operator()(const std::array<double, 2>& x) const
  {
    return _interpolant.interpolate(_Order, x);
  }
};


// f = cos(x) sin(y)
void
func(const std::array<double, 2>& x,
     numerical::ValueGradientHessian<double>* f)
{
  f->f = std::cos(x[0]) * std::sin(x[1]);
  f->fx = - std::sin(x[0]) * std::sin(x[1]);
  f->fy = std::cos(x[0]) * std::cos(x[1]);
  f->fxx = - std::cos(x[0]) * std::sin(x[1]);
  f->fxy = - std::sin(x[0]) * std::cos(x[1]);
  f->fyy = - std::cos(x[0]) * std::sin(x[1]);
}

int
main()
{
  //
  // Simple test.
  //
  {
    // Define types.
    // CONTINUE
    //typedef numerical::QuinticInterpolation2D<double> Interp;
    typedef QuinticInterpolation2DTest<double> Interp;
    typedef Interp::ValueGradientHessian ValueGradientHessian;
    typedef Interp::Point Point;
    typedef Interp::BBox BBox;
    typedef Interp::ValueGrid ValueGrid;
    typedef ValueGrid::SizeList SizeList;
    typedef ValueGrid::IndexList IndexList;
    typedef ValueGrid::Index Index;

    // The grid has 3 x 3 cells. (4 x 4 grid points.)
    const SizeList Extents = {{4, 4}};
    const Point Lo = {{ -1, -1}};
    const Point Hi = {{2, 2}};
    const Point Delta = (Hi - Lo) /
      ext::convert_array<double>(Extents - std::size_t(1));
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    // Sample the function.
    IndexList i;
    Point x;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        x = Lo + stlib::ext::convert_array<double>(i) * Delta;
        valueGrid(i) = x[0] * x[0] + x[1] * x[1];
      }
    }

    // Construct the interpolating function.
    Interp interp(valueGrid, Domain);

    // f = x^2 + y^2
    // f_x = 2 x + y^2
    // f_xx = 2 + y^2

    // f f_x f_xx
    // 0 0 2
    // 0 0 0
    // 0 0 0

    //std::cerr << "_grid(1, 1) = " << interp._grid(1, 1) << '\n';

    // Evaluate the interpolant at (0, 0).
    ValueGradientHessian y;
    x[0] = 0;
    x[1] = 0;
    interp(x, &y);
    // CONTINUE
    //std::cerr << y << '\n';
  }




  //
  // Example code.
  //
  {
    // Define types.
    // CONTINUE
    //typedef numerical::QuinticInterpolation2D<double> Interp;
    typedef QuinticInterpolation2DTest<double> Interp;
    typedef Interp::ValueGradientHessian ValueGradientHessian;
    typedef Interp::Point Point;
    typedef Interp::BBox BBox;
    typedef Interp::ValueGrid ValueGrid;
    typedef ValueGrid::SizeList SizeList;
    typedef ValueGrid::IndexList IndexList;
    typedef ValueGrid::Index Index;

    // The grid has 100 x 100 cells. (101 x 101 grid points.)
    const SizeList Extents = {{101, 101}};
    const double Pi = numerical::Constants<double>::Pi();
    const Point Lo = {{0, 0}};
    const Point Hi = {{Pi, Pi}};
    const Point Delta = (Hi - Lo) /
      ext::convert_array<double>(Extents - std::size_t(1));
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    // Sample the function.
    IndexList i;
    Point x;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        x = stlib::ext::convert_array<double>(i) * Delta;
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
      }
    }

    // Construct the interpolating function.
    Interp interp(valueGrid, Domain);

    // Evaluate the interpolant at (pi/3, pi/3).
    ValueGradientHessian y;
    x[0] = Pi / 3;
    x[1] = Pi / 3;

    // CONTINUE
    func(x, &y);
    //std::cerr << y << '\n';

    interp(x, &y);

    // Check the accuracy of the interpolant.
    const double Dx = stlib::ext::max(Delta);
    // CONTINUE
    //std::cerr << y << '\n';
    assert(std::abs(y.f - std::cos(x[0]) * std::sin(x[1])) < Dx * Dx * Dx);
    assert(std::abs(y.fx + std::sin(x[0]) * std::sin(x[1])) < Dx * Dx);
    assert(std::abs(y.fy - std::cos(x[0]) * std::cos(x[1])) < Dx * Dx);
    assert(std::abs(y.fxx + std::cos(x[0]) * std::sin(x[1])) < Dx);
    assert(std::abs(y.fxy + std::sin(x[0]) * std::cos(x[1])) < Dx);
    assert(std::abs(y.fyy + std::cos(x[0]) * std::sin(x[1])) < Dx);
  }
  {
    // Define types.
    // CONTINUE
    //typedef numerical::QuinticInterpolation2D<double, true> Periodic;
    typedef QuinticInterpolation2DTest<double, true> Periodic;
    typedef Periodic::ValueGradientHessian ValueGradientHessian;
    typedef Periodic::Point Point;
    typedef Periodic::BBox BBox;
    typedef Periodic::ValueGrid ValueGrid;
    typedef ValueGrid::SizeList SizeList;
    typedef ValueGrid::IndexList IndexList;
    typedef ValueGrid::Index Index;

    // The grid has 100 x 100 cells. (100 x 100 grid points.)
    const SizeList Extents = {{100, 100}};
    const double Pi = numerical::Constants<double>::Pi();
    const Point Lo = {{0, 0}};
    const Point Hi = {{2 * Pi, 2 * Pi}};
    const Point Delta = (Hi - Lo) / ext::convert_array<double>(Extents);
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    // Sample the function.
    IndexList i;
    Point x;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        x = stlib::ext::convert_array<double>(i) * Delta;
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
      }
    }

    // Construct the interpolating function.
    Periodic interp(valueGrid, Domain);

    // Evaluate the interpolant at (pi/3, pi/3).
    ValueGradientHessian y;
    x[0] = Pi / 3;
    x[1] = Pi / 3;

    // 0 0 0
    // 1 0 -1
    // 0 0 0
    //
    // 0 0 0
    // 0.0314108 -1.16245e-05 -7.74966e-06
    // 1.54993e-05 -7.74966e-06 0
    // f f_x f_xx
    //
    //
    //std::cerr << "_grid(0, 0) = " << interp._grid(0, 0) << '\n';
    func(x, &y);
    //std::cerr << y << '\n';

    interp(x, &y);
    // CONTINUE
    //std::cerr << y << '\n';

    // Check the accuracy of the interpolant.
    const double Dx = stlib::ext::max(Delta);
    assert(std::abs(y.f - std::cos(x[0]) * std::sin(x[1])) < Dx * Dx * Dx);
    assert(std::abs(y.fx + std::sin(x[0]) * std::sin(x[1])) < Dx * Dx);
    assert(std::abs(y.fy - std::cos(x[0]) * std::cos(x[1])) < Dx * Dx);
    assert(std::abs(y.fxx + std::cos(x[0]) * std::sin(x[1])) < Dx);
    assert(std::abs(y.fxy + std::sin(x[0]) * std::cos(x[1])) < Dx);
    assert(std::abs(y.fyy + std::cos(x[0]) * std::sin(x[1])) < Dx);
  }

  typedef QuinticInterpolation2DTest<double> F;
  typedef F::ValueGradientHessian ValueGradientHessian;
  typedef F::Dt2d3 Dt2d3;
  typedef F::ValueGrid ValueGrid;
  typedef F::Point Point;
  typedef F::BBox BBox;
  typedef ValueGrid::SizeList SizeList;
  typedef ValueGrid::IndexList IndexList;
  typedef ValueGrid::Index Index;

  const double Eps = std::numeric_limits<double>::epsilon();

  //-------------------------------------------------------------------------
  // Periodic, constant.
  //-------------------------------------------------------------------------

  {
    SizeList Extents = {{1, 1}};
    const double Value = 1;
    const ValueGradientHessian Vgh = {Value, 0, 0, 0, 0, 0};
    ValueGrid valueGrid(Extents, Value);
    const BBox Domain = {{{0., 0.}}, {{1., 1.}}};
    QuinticInterpolation2DTest<double, true> f(valueGrid, Domain);

    // Check the grid values.
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        const Dt2d3& x = f._grid(i);
        assert(x(0, 0) == Value);
        assert(x(1, 0) == 0);
        assert(x(2, 0) == 0);
        assert(x(0, 1) == 0);
        assert(x(1, 1) == 0);
        assert(x(2, 1) == 0);
        assert(x(0, 2) == 0);
        assert(x(1, 2) == 0);
        assert(x(2, 2) == 0);
      }
    }

    // Check the values at the grid points.
    {
      Point x = {{0, 0}};
      ValueGradientHessian y;
      f(x, &y);
      assert(areEqual(y, Vgh));
      assert(numerical::areEqual(f.interpolate<0>(x), Value));
      assert(numerical::areEqual(f.interpolate<1>(x), Value));
      assert(numerical::areEqual(f.interpolate<3>(x), Value));
      assert(numerical::areEqual(f.interpolate<5>(x), Value));
      assert(numerical::areEqual(f.interpolate(0, x), Value));
      assert(numerical::areEqual(f.interpolate(1, x), Value));
      assert(numerical::areEqual(f.interpolate(3, x), Value));
      assert(numerical::areEqual(f.interpolate(5, x), Value));
      Point gradient;
      assert(numerical::areEqual(f.interpolate<3>(x, &gradient), Value));
      assert(numerical::areEqual(gradient, Point{{0., 0.}}));
      assert(numerical::areEqual(f.interpolate<5>(x, &gradient), Value));
      assert(numerical::areEqual(gradient, Point{{0., 0.}}));
      assert(numerical::areEqual(f.interpolate(3, x, &gradient), Value));
      assert(numerical::areEqual(gradient, Point{{0., 0.}}));
      assert(numerical::areEqual(f.interpolate(5, x, &gradient), Value));
      assert(numerical::areEqual(gradient, Point{{0., 0.}}));
    }
    {
      Point x = {{1 - Eps, 0}};
      ValueGradientHessian y;
      f(x, &y);
      assert(areEqual(y, Vgh));
      assert(numerical::areEqual(f.interpolate<0>(x), Value));
      assert(numerical::areEqual(f.interpolate<1>(x), Value));
      assert(numerical::areEqual(f.interpolate<3>(x), Value));
      assert(numerical::areEqual(f.interpolate<5>(x), Value));
      assert(numerical::areEqual(f.interpolate(0, x), Value));
      assert(numerical::areEqual(f.interpolate(1, x), Value));
      assert(numerical::areEqual(f.interpolate(3, x), Value));
      assert(numerical::areEqual(f.interpolate(5, x), Value));
    }
    {
      Point x = {{0, 1 - Eps}};
      ValueGradientHessian y;
      f(x, &y);
      assert(areEqual(y, Vgh));
      assert(numerical::areEqual(f.interpolate<0>(x), Value));
      assert(numerical::areEqual(f.interpolate<1>(x), Value));
      assert(numerical::areEqual(f.interpolate<3>(x), Value));
      assert(numerical::areEqual(f.interpolate<5>(x), Value));
      assert(numerical::areEqual(f.interpolate(0, x), Value));
      assert(numerical::areEqual(f.interpolate(1, x), Value));
      assert(numerical::areEqual(f.interpolate(3, x), Value));
      assert(numerical::areEqual(f.interpolate(5, x), Value));
    }
    {
      Point x = {{1 - Eps, 1 - Eps}};
      ValueGradientHessian y;
      f(x, &y);
      assert(areEqual(y, Vgh));
      assert(numerical::areEqual(f.interpolate<0>(x), Value));
      assert(numerical::areEqual(f.interpolate<1>(x), Value));
      assert(numerical::areEqual(f.interpolate<3>(x), Value));
      assert(numerical::areEqual(f.interpolate<5>(x), Value));
      assert(numerical::areEqual(f.interpolate(0, x), Value));
      assert(numerical::areEqual(f.interpolate(1, x), Value));
      assert(numerical::areEqual(f.interpolate(3, x), Value));
      assert(numerical::areEqual(f.interpolate(5, x), Value));
    }
    // Check the values in the interior.
    {
      ValueGradientHessian a;
      ValueGradientHessian b = Vgh;
      {
        Point x = {{0.5, 0}};
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
      }
      {
        Point x = {{0, 0.5}};
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
      }
      {
        Point x = {{0.5, 1 - Eps}};
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
      }
      {
        Point x = {{1 - Eps, 0.5}};
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
      }
      {
        Point x = {{0.5, 0.5}};
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
      }
    }
    // Check the periodicity.
    {
      std::vector<Point> points;
      points.push_back(Point{{0., 0.}});
      points.push_back(Point{{1., 1.}});
      points.push_back(Point{{0.5, 0.}});
      points.push_back(Point{{0., 0.5}});
      points.push_back(Point{{0.5, 0.5}});
      for (std::size_t i = 0; i != points.size(); ++i) {
        const Point x = points[i];
        {
          const Point y = x + Point{{2., 3.}};
          assert(numerical::areEqual(f.interpolate<0>(x),
                                     f.interpolate<0>(y), 10));
          assert(numerical::areEqual(f.interpolate<1>(x),
                                     f.interpolate<1>(y), 10));
          assert(numerical::areEqual(f.interpolate<3>(x),
                                     f.interpolate<3>(y), 10));
          assert(numerical::areEqual(f.interpolate<5>(x),
                                     f.interpolate<5>(y), 10));
        }
        {
          const Point y = x - Point{{2., 3.}};
          assert(numerical::areEqual(f.interpolate<0>(x),
                                     f.interpolate<0>(y), 10));
          assert(numerical::areEqual(f.interpolate<1>(x),
                                     f.interpolate<1>(y), 10));
          assert(numerical::areEqual(f.interpolate<3>(x),
                                     f.interpolate<3>(y), 10));
          assert(numerical::areEqual(f.interpolate<5>(x),
                                     f.interpolate<5>(y), 10));
        }
      }
    }
  }

  //-------------------------------------------------------------------------
  // Test with a linear function.
  //-------------------------------------------------------------------------

  //
  // Use the linear function x + 2 y on a 1x1 array of cells that covers
  // the unit square.
  //
  {
    SizeList Extents = {{2, 2}};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        valueGrid(i) = i[0] + 2 * i[1];
      }
    }

    const BBox Domain = {{{0., 0.}}, {{1., 1.}}};
    F f(valueGrid, Domain);

    // Check the grid values.
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        const Dt2d3& x = f._grid(i);
        assert(x(0, 0) == valueGrid(i));
        assert(x(1, 0) == 1);
        assert(x(2, 0) == 0);
        assert(x(0, 1) == 2);
        assert(x(1, 1) == 0);
        assert(x(2, 1) == 0);
        assert(x(0, 2) == 0);
        assert(x(1, 2) == 0);
        assert(x(2, 2) == 0);
      }
    }

    // Check the values at the grid points.
    {
      IndexList i = {{0, 0}};
      Point x = {{0, 0}};
      ValueGradientHessian y;
      f(x, &y);
      // CONTINUE
      //assert(areEqual(y, f._grid(i)));
      assert(numerical::areEqual(f.interpolate<0>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<1>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<3>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<5>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate(0, x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate(1, x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate(3, x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate(5, x), f._grid(i)(0, 0)));
      Point g;
      assert(numerical::areEqual(f.interpolate<3>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
      assert(numerical::areEqual(f.interpolate<5>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
    }
    {
      IndexList i = {{1, 0}};
      Point x = {{1 - Eps, 0}};
      ValueGradientHessian y;
      f(x, &y);
      // CONTINUE
      //assert(areEqual(y, f._grid(i)));
      assert(numerical::areEqual(f.interpolate<0>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<1>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<3>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<5>(x), f._grid(i)(0, 0)));
      Point g;
      assert(numerical::areEqual(f.interpolate<3>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
      assert(numerical::areEqual(f.interpolate<5>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
    }
    {
      IndexList i = {{0, 1}};
      Point x = {{0, 1 - Eps}};
      ValueGradientHessian y;
      f(x, &y);
      // CONTINUE
      //assert(areEqual(y, f._grid(i)));
      assert(numerical::areEqual(f.interpolate<0>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<1>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<3>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<5>(x), f._grid(i)(0, 0)));
      Point g;
      assert(numerical::areEqual(f.interpolate<3>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
      assert(numerical::areEqual(f.interpolate<5>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0)));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1)));
    }
    {
      IndexList i = {{1, 1}};
      Point x = {{1 - Eps, 1 - Eps}};
      ValueGradientHessian y;
      f(x, &y);
      // CONTINUE
      //assert(areEqual(y, f._grid(i)));
      assert(numerical::areEqual(f.interpolate<0>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<1>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<3>(x), f._grid(i)(0, 0)));
      assert(numerical::areEqual(f.interpolate<5>(x), f._grid(i)(0, 0), 10));
      Point g;
      assert(numerical::areEqual(f.interpolate<3>(x, &g), f._grid(i)(0, 0)));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0), 10));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1), 10));
      assert(numerical::areEqual(f.interpolate<5>(x, &g), f._grid(i)(0, 0), 10));
      assert(numerical::areEqual(g[0], f._grid(i)(1, 0), 10));
      assert(numerical::areEqual(g[1], f._grid(i)(0, 1), 100));
    }
    // Check the values in the interior.
    {
      ValueGradientHessian a;
      ValueGradientHessian b = {0, 1, 2, 0, 0, 0};
      {
        Point x = {{0.5, 0}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), 1, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{0, 0.5}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), 2, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{0.5, 1 - Eps}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), 3, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{1 - Eps, 0.5}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), 3, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e4));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{0.5, 0.5}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), 3, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
    }
  }

  //
  // Use the linear function x + 2 y on a 1x1 array of cells that covers
  // the domain with corners (2, 3) and (5, 7).
  //
  {
    SizeList Extents = {{2, 2}};
    const Point Lo = {{2, 3}};
    const Point Hi = {{5, 7}};
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        Point x;
        for (std::size_t n = 0; n != x.size(); ++n) {
          x[n] = (Hi[n] - Lo[n]) * i[n] / double(Extents[n] - 1) + Lo[n];
        }
        valueGrid(i) = x[0] + 2 * x[1];
      }
    }

    F f(valueGrid, Domain);

    // Check the grid values.
    {
      const Point s = f._inverseWidths;
      for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
        for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
          const Dt2d3& x = f._grid(i);
          assert(x(0, 0) == valueGrid(i));
          assert(x(1, 0) * s[0] == 1);
          assert(x(2, 0) == 0);
          assert(x(0, 1) * s[1] == 2);
          assert(x(1, 1) == 0);
          assert(x(2, 1) == 0);
          assert(x(0, 2) == 0);
          assert(x(1, 2) == 0);
          assert(x(2, 2) == 0);
        }
      }
    }

    {
      ValueGradientHessian a;
      ValueGradientHessian b = {0, 1, 2, 0, 0, 0};
      // Check the values at the grid points.
      {
        Point x = Lo;
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), Lo[1]}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b, 1e9));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Lo[0], Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b, 1e9));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      // Check the values in the interior.
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), Lo[1]}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Lo[0], 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b, 1e9));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b, 1e9));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 1e5));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
    }
  }

  //
  // Use the linear function x + 2 y on a 2x3 array of cells that covers
  // the domain with corners (2, 3) and (5, 7).
  //
  {
    SizeList Extents = {{3, 4}};
    const Point Lo = {{2, 3}};
    const Point Hi = {{5, 7}};
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        Point x;
        for (std::size_t n = 0; n != x.size(); ++n) {
          x[n] = (Hi[n] - Lo[n]) * i[n] / double(Extents[n] - 1) + Lo[n];
        }
        valueGrid(i) = x[0] + 2 * x[1];
      }
    }

    F f(valueGrid, Domain);

    // Check the grid values.
    {
      const Point s = f._inverseWidths;
      for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
        for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
          const Dt2d3& x = f._grid(i);
          assert(numerical::areEqual(x(0, 0), valueGrid(i)));
          assert(numerical::areEqual(x(1, 0) * s[0], 1));
          assert(numerical::areEqual(x(2, 0), 0));
          assert(numerical::areEqual(x(0, 1) * s[1], 2));
          assert(numerical::areEqual(x(1, 1), 0));
          assert(numerical::areEqual(x(2, 1), 0));
          assert(numerical::areEqual(x(0, 2), 0, 10));
          assert(numerical::areEqual(x(1, 2), 0));
          assert(numerical::areEqual(x(2, 2), 0, 10));
        }
      }
    }

    {
      ValueGradientHessian a;
      ValueGradientHessian b = {0, 1, 2, 0, 0, 0};
      // Check the values at the grid points.
      {
        Point x = Lo;
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), Lo[1]}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Lo[0], Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<0>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(0, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(1, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(3, x), b.f, 10));
        assert(numerical::areEqual(f.interpolate(5, x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
      // Check the values in the interior.
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), Lo[1]}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
      }
      {
        Point x = {{Lo[0], 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), Hi[1]* (1 - Eps)}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
      {
        Point x = {{Hi[0]* (1 - Eps), 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
      {
        Point x = {{0.5 * (Lo[0] + Hi[0]), 0.5 * (Lo[1] + Hi[1])}};
        b.f = b.fx * x[0] + b.fy * x[1];
        f(x, &a);
        assert(areEqual(a, b));
        assert(numerical::areEqual(f.interpolate<1>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<3>(x), b.f, 10));
        assert(numerical::areEqual(f.interpolate<5>(x), b.f, 10));
        Point g;
        assert(numerical::areEqual(f.interpolate<3>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 10));
        assert(numerical::areEqual(f.interpolate<5>(x, &g), b.f, 10));
        assert(numerical::areEqual(g[0], b.fx, 10));
        assert(numerical::areEqual(g[1], b.fy, 100));
      }
    }
  }

  //-------------------------------------------------------------------------
  // Test with a quadratic function.
  //-------------------------------------------------------------------------

  //
  // Use the function x^2 + 2 x y + y^2 on a 3x3 array of cells that covers
  // the unit square.
  //
  {
    SizeList Extents = {{4, 4}};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        const Point x = (1. / 3) * stlib::ext::convert_array<double>(i);
        valueGrid(i) = x[0] * x[0] + 2 * x[0] * x[1] + x[1] * x[1];
      }
    }

    const BBox Domain = {{{0., 0.}}, {{1., 1.}}};
    F f(valueGrid, Domain);

    std::cout << "\nQuadratic function: x^2 + 2xy + y^2\n";
    Point g1, g2;
    {
      InterpolantValue<F, 5> iv(f);
      std::array<double, 2> maxError = {{0, 0}};
      for (std::size_t i = 1; i != 10; ++i) {
        for (std::size_t j = 1; j != 10; ++j) {
          const Point x = {{0.1 * i, 0.1 * j}};
          f.interpolate<5>(x, &g1);
          numerical::gradientRidders(iv, x, &g2);
          for (std::size_t k = 0; k != maxError.size(); ++k) {
            maxError[k] = std::max(maxError[k],
                                   std::abs(g1[k] - g2[k]));
          }
        }
      }
      std::cout << "Quintic, max error in derivative = " << maxError << '\n';
    }
    {
      InterpolantValue<F, 3> iv(f);
      std::array<double, 2> maxError = {{0, 0}};
      for (std::size_t i = 1; i != 10; ++i) {
        for (std::size_t j = 1; j != 10; ++j) {
          const Point x = {{0.1 * i, 0.1 * j}};
          f.interpolate<3>(x, &g1);
          numerical::gradientRidders(iv, x, &g2);
          for (std::size_t k = 0; k != maxError.size(); ++k) {
            maxError[k] = std::max(maxError[k],
                                   std::abs(g1[k] - g2[k]));
          }
        }
      }
      std::cout << "Cubic, max error in derivative = " << maxError << '\n';
    }
  }

  //-------------------------------------------------------------------------
  // Test with a cubic function.
  //-------------------------------------------------------------------------

  //
  // Use the function (x + 2 * y)^3 on a 3x3 array of cells that covers
  // the unit square.
  //
  {
    SizeList Extents = {{4, 4}};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        const Point x = (1. / 3) * stlib::ext::convert_array<double>(i);
        valueGrid(i) = (x[0] + 2 * x[1]) * (x[0] + 2 * x[1]) *
                       (x[0] + 2 * x[1]);
      }
    }

    const BBox Domain = {{{0., 0.}}, {{1., 1.}}};
    F f(valueGrid, Domain);

    std::cout << "\nCubic function: (x + 2y)^3\n";
    Point g1, g2;
    {
      InterpolantValue<F, 5> iv(f);
      std::array<double, 2> maxError = {{0, 0}};
      for (std::size_t i = 1; i != 10; ++i) {
        for (std::size_t j = 1; j != 10; ++j) {
          const Point x = {{0.1 * i, 0.1 * j}};
          f.interpolate<5>(x, &g1);
          numerical::gradientRidders(iv, x, &g2);
          for (std::size_t k = 0; k != maxError.size(); ++k) {
            maxError[k] = std::max(maxError[k],
                                   std::abs(g1[k] - g2[k]));
          }
        }
      }
      std::cout << "Quintic, max error in derivative = " << maxError << '\n';
    }
    {
      InterpolantValue<F, 3> iv(f);
      std::array<double, 2> maxError = {{0, 0}};
      for (std::size_t i = 1; i != 10; ++i) {
        for (std::size_t j = 1; j != 10; ++j) {
          const Point x = {{0.1 * i, 0.1 * j}};
          f.interpolate<3>(x, &g1);
          numerical::gradientRidders(iv, x, &g2);
          for (std::size_t k = 0; k != maxError.size(); ++k) {
            maxError[k] = std::max(maxError[k],
                                   std::abs(g1[k] - g2[k]));
          }
        }
      }
      std::cout << "Cubic, max error in derivative = " << maxError << '\n';
    }
  }

  //-------------------------------------------------------------------------
  // Next use f = cos(x) sin(y).
  //-------------------------------------------------------------------------
  //
  // Use the function cos(x) sin(y) on a 100x100 array of cells that covers
  // the domain with corners (0, 0) and (pi, pi).
  //
  {
    const SizeList Extents = {{101, 101}};
    const double Pi = numerical::Constants<double>::Pi();
    const Point Lo = {{0, 0}};
    const Point Hi = {{Pi, Pi}};
    const double Dx = (Hi[0] - Lo[0]) / (Extents[0] - 1);
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        Point x;
        for (std::size_t n = 0; n != x.size(); ++n) {
          x[n] = (Hi[n] - Lo[n]) * i[n] / double(Extents[n] - 1) + Lo[n];
        }
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
      }
    }

    F f(valueGrid, Domain);
    InterpolantValue<F, 3> cubic(f);
    InterpolantValue<F, 5> quintic(f);

    // Test only the interior. The missing derivatives at the boundary
    // increase the error.
    std::cout << "\nUse a 200 * 200 array of cells.\n";
    ValueGradientHessian a, b, m;
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    std::array<IndexList, 6> mi;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    const double Delta = std::pow(std::numeric_limits<double>::epsilon(),
                                  0.25);
    double maxNumericalGradientError3 = 0;
    double maxNumericalGradientError5 = 0;
    double maxCenteredGradientError = 0;
    double d, y;
    Point gradient, g;
    for (i[0] = 2; i[0] != Index(199); ++i[0]) {
      for (i[1] = 2; i[1] != Index(199); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 200.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 200.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx * Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx * Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx * Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }

        // Other interfaces.

        y = f.interpolate<1>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<3>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);

        y = f.interpolate<3>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        // Compare with the numerical gradient of the cubic interpolant.
        numerical::gradientRidders(cubic, x, &gradient);
        d = std::abs(g[0] - gradient[0]);
        maxNumericalGradientError3 =
          std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        maxNumericalGradientError3 =
          std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);
        // Compare with the centered difference gradient of the cubic
        // interpolant.
        gradient[0] = (cubic(x + Point{{Delta, 0.}}) -
                       cubic(x - Point{{Delta, 0.}})) / (2 * Delta);
        maxCenteredGradientError = std::max(maxCenteredGradientError,
                                            std::abs(g[0] - gradient[0]));
        gradient[1] = (cubic(x + Point{{0., Delta}}) -
                       cubic(x - Point{{0., Delta}})) / (2 * Delta);
        maxCenteredGradientError = std::max(maxCenteredGradientError,
                                            std::abs(g[1] - gradient[1]));

        y = f.interpolate<5>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - gradient[0]);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        assert(d < Dx * Dx);
        // Compare with the numerical gradient of the quintic interpolant.
        numerical::gradientRidders(quintic, x, &gradient);
        d = std::abs(g[0] - gradient[0]);
        maxNumericalGradientError5 =
          std::max(d, maxNumericalGradientError5);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        maxNumericalGradientError5 =
          std::max(d, maxNumericalGradientError5);
        assert(d < Dx * Dx);
      }
    }
    std::cout << "Maximum errors over the interior cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }
    std::cout << "maxNumericalGradientError3 = "
              << maxNumericalGradientError3 << '\n'
              << "maxCenteredGradientError = "
              << maxCenteredGradientError << '\n'
              << "maxNumericalGradientError5 = "
              << maxNumericalGradientError5 << '\n';

    // Test the whole grid.
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    for (i[0] = 0; i[0] != Index(200); ++i[0]) {
      for (i[1] = 0; i[1] != Index(200); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 200.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 200.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < 2 * Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < 40 * Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        // CONTINUE HERE
        assert(d < 40 * Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }

        // Other interfaces.

        y = f.interpolate<1>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<3>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x);
        d = std::abs(y - b.f);
        assert(d < 2 * Dx * Dx * Dx);

        y = f.interpolate<3>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < 20 * Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x, &g);
        d = std::abs(y - b.f);
        assert(d < 2 * Dx * Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < 20 * Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }



    // Test only the interior. The missing derivatives at the boundary
    // increase the error.
    std::cout << "\nUse a 300 * 300 array of cells.\n";
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    maxNumericalGradientError3 = 0;
    for (i[0] = 3; i[0] != Index(298); ++i[0]) {
      for (i[1] = 3; i[1] != Index(298); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 300.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 300.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }

        // Other interfaces.

        y = f.interpolate<1>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<3>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);

        y = f.interpolate<3>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        // Compare with the numerical gradient of the cubic interpolant.
        numerical::gradientRidders(cubic, x, &gradient);
        d = std::abs(g[0] - gradient[0]);
        maxNumericalGradientError3 = std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        maxNumericalGradientError3 = std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - gradient[0]);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        assert(d < Dx * Dx);
      }
    }
    std::cout << "Maximum errors over the interior cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }
    std::cout << "maxNumericalGradientError3 = " << maxNumericalGradientError3
              << '\n';

    // Test the whole grid.
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    for (i[0] = 0; i[0] != Index(300); ++i[0]) {
      for (i[1] = 0; i[1] != Index(300); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 300.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 300.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < 5 * Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < 20 * Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < 100 * Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < 1.1 * Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < 1.1 * Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }



    // Test only the interior. The missing derivatives at the boundary
    // increase the error.
    std::cout << "\nUse a 1000 * 1000 array of cells.\n";
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    maxNumericalGradientError3 = 0;
    for (i[0] = 10; i[0] != Index(991); ++i[0]) {
      for (i[1] = 10; i[1] != Index(991); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 1000.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 1000.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }

        // Other interfaces.

        y = f.interpolate<1>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<3>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);

        y = f.interpolate<3>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        // Compare with the numerical gradient of the cubic interpolant.
        numerical::gradientRidders(cubic, x, &gradient);
        d = std::abs(g[0] - gradient[0]);
        maxNumericalGradientError3 = std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        maxNumericalGradientError3 = std::max(d, maxNumericalGradientError3);
        assert(d < Dx * Dx);

        y = f.interpolate<5>(x, &g);
        d = std::abs(y - b.f);
        assert(d < Dx * Dx * Dx);
        d = std::abs(g[0] - b.fx);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - b.fy);
        assert(d < Dx * Dx);
        d = std::abs(g[0] - gradient[0]);
        assert(d < Dx * Dx);
        d = std::abs(g[1] - gradient[1]);
        assert(d < Dx * Dx);
      }
    }
    std::cout << "Maximum errors over the interior cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }
    std::cout << "maxNumericalGradientError3 = "
              << maxNumericalGradientError3 << '\n';

    // Test the whole grid.
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    for (i[0] = 0; i[0] != Index(1000); ++i[0]) {
      for (i[1] = 0; i[1] != Index(1000); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 1000.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 1000.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < 5 * Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < 50 * Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < 100 * Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < 2 * Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < 2 * Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }
  }

  //-------------------------------------------------------------------------
  // Periodic, f = cos(x) sin(y).
  //-------------------------------------------------------------------------
  //
  // Use the function cos(x) sin(y) on a 100x100 array of cells that covers
  // the domain with corners (0, 0) and (2 pi, 2 pi).
  //
  {
    const SizeList Extents = {{100, 100}};
    const double Pi = numerical::Constants<double>::Pi();
    const Point Lo = {{0, 0}};
    const Point Hi = {{2 * Pi, 2 * Pi}};
    const double Dx = (Hi[0] - Lo[0]) / Extents[0];
    const BBox Domain = {Lo, Hi};
    ValueGrid valueGrid(Extents);
    IndexList i;
    for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
      for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        Point x;
        for (std::size_t n = 0; n != x.size(); ++n) {
          x[n] = (Hi[n] - Lo[n]) * i[n] / double(Extents[n]) + Lo[n];
        }
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
      }
    }

    numerical::QuinticInterpolation2D<double, true> f(valueGrid, Domain);

    // Test all grid points.
    std::cout << "\nPeriodic. Use a 200 * 200 array of cells.\n";
    ValueGradientHessian a, b, m;
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    std::array<IndexList, 6> mi;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    double d;
    for (i[0] = 0; i[0] != Index(200); ++i[0]) {
      for (i[1] = 0; i[1] != Index(200); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 200.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 200.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx * Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx * Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx * Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }

    // Test all points.
    std::cout << "\nPeriodic. Use a 300 * 300 array of cells.\n";
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    for (i[0] = 0; i[0] != Index(300); ++i[0]) {
      for (i[1] = 0; i[1] != Index(300); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 300.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 300.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }

    // Test all points.
    std::cout << "\nPeriodic. Use a 1000 * 1000 array of cells.\n";
    m.f = m.fx = m.fy = m.fxx = m.fxy = m.fyy = 0;
    for (std::size_t n = 0; n != mi.size(); ++n) {
      mi[n][0] = 0;
      mi[n][1] = 0;
    }
    for (i[0] = 0; i[0] != Index(1000); ++i[0]) {
      for (i[1] = 0; i[1] != Index(1000); ++i[1]) {
        Point x = {{
            Lo[0] + (Hi[0] - Lo[0])* i[0] / 1000.,
            Lo[1] + (Hi[1] - Lo[1])* i[1] / 1000.
          }
        };
        f(x, &a);
        func(x, &b);

        d = std::abs(a.f - b.f);
        assert(d < Dx * Dx * Dx);
        if (d > m.f) {
          m.f = d;
          mi[0] = i;
        }

        d = std::abs(a.fx - b.fx);
        assert(d < Dx * Dx);
        if (d > m.fx) {
          m.fx = d;
          mi[1] = i;
        }

        d = std::abs(a.fy - b.fy);
        assert(d < Dx * Dx);
        if (d > m.fy) {
          m.fy = d;
          mi[2] = i;
        }

        d = std::abs(a.fxx - b.fxx);
        assert(d < Dx);
        if (d > m.fxx) {
          m.fxx = d;
          mi[3] = i;
        }

        d = std::abs(a.fxy - b.fxy);
        assert(d < Dx);
        if (d > m.fxy) {
          m.fxy = d;
          mi[4] = i;
        }

        d = std::abs(a.fyy - b.fyy);
        assert(d < Dx);
        if (d > m.fyy) {
          m.fyy = d;
          mi[5] = i;
        }
      }
    }
    std::cout << "Maximum errors over all cells:\n" << m << '\n';
    for (std::size_t n = 0; n != mi.size(); ++n) {
      std::cout << "Location " << n << " = " << mi[n] << '\n';
    }
  }

  return 0;
}

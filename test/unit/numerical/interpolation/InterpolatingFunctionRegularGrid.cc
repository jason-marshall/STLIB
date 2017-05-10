// -*- C++ -*-

#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGrid.h"
#include "stlib/numerical/equality.h"
#include "stlib/numerical/constants.h"

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  //
  // 1-D, constant interpolation, plain.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 0;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;

    {
      SizeList extents = {{1}};
      Grid data(extents);
      data[0] = 2;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);

      // Copy constructor.
      {
        InterpolatingFunction g = f;
      }
      // Assignment operator
      {
        InterpolatingFunction g = f;
        g = f;
      }

      Point x = {{0}};
      assert(numerical::areEqual(f(x), data[0]));
      IndexList i;
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 10;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));
    }
    {
      SizeList extents = {{2}};
      Grid data(extents);
      data[0] = 2;
      data[1] = 3;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);
      Point x = {{0}};
      assert(numerical::areEqual(f(x), data[0]));
      IndexList i;
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5.99;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 6.01;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 10;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));
    }
    {
      SizeList extents = {{3}};
      Grid data(extents);
      data[0] = 2;
      data[1] = 3;
      data[2] = 5;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);

      Point x = {{0}};
      assert(numerical::areEqual(f(x), data[0]));
      assert(f.interpolate<0>(x) == f(x));
      assert(f.interpolate(0, x) == f(x));
      IndexList i;
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5.49;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5.51;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 6.49;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 6.51;
      assert(numerical::areEqual(f(x), data[2]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{2}}));

      x[0] = 10;
      assert(numerical::areEqual(f(x), data[2]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{2}}));
    }
  }
  //
  // 1-D, constant interpolation, periodic.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 0;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;

    {
      SizeList extents = {{1}};
      Grid data(extents);
      data[0] = 2;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);

      Point x = {{0}};
      assert(numerical::areEqual(f(x), data[0]));
      IndexList i;
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 10;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));
    }
    {
      SizeList extents = {{2}};
      Grid data(extents);
      data[0] = 2;
      data[1] = 3;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);

      Point x = {{3}};
      assert(numerical::areEqual(f(x), data[0]));
      IndexList i;
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5.49;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 5.51;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 6.49;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 6.51;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 7.49;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));

      x[0] = 7.51;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 8.49;
      assert(numerical::areEqual(f(x), data[1]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{1}}));

      x[0] = 8.51;
      assert(numerical::areEqual(f(x), data[0]));
      f.snapToGrid(x, &i);
      assert(i == (IndexList{{0}}));
    }
  }
  //
  // 1-D, linear interpolation, plain.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;

    SizeList extents = {{2}};
    Grid data(extents);
    data[0] = 2;
    data[1] = 3;
    BBox domain = {{{5.}}, {{7.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(domain.lower), data[0]));
    assert(numerical::areEqual(f(domain.upper), data[1]));
  }
  //
  // 1-D, linear interpolation, periodic.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;

    SizeList extents = {{1}};
    Grid data(extents);
    data[0] = 2;
    BBox domain = {{{5.}}, {{7.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(domain.lower), data[0]));
    assert(numerical::areEqual(f(domain.upper), data[0]));
  }
  //
  // 2-D, constant interpolation, plain.
  //
  {
    const std::size_t Dimension = 2;
    const std::size_t Order = 0;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    // Make an array. Each element value is the sum of its index list.
    SizeList extents = {{2, 2}};
    Grid data(extents);
    const Iterator end = Iterator::end(data.range());
    for (Iterator i = Iterator::begin(data.range()); i != end; ++i) {
      data(*i) = stlib::ext::sum(*i);
    }
    BBox domain = {{{2., 3.}}, {{5., 7.}}};
    InterpolatingFunction f(data, domain);
    IndexList i, index;
    Point x;

    index = IndexList{{0, 0}};
    x = Point{{2., 3.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{0., 3.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{2., 0.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{3.49, 4.99}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);

    index = IndexList{{0, 1}};
    x = Point{{2., 7.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{0., 7.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{2., 10.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{3.49, 5.01}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);

    index = IndexList{{1, 0}};
    x = Point{{5., 3.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{10., 3.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{5., 0.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{3.51, 4.99}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);

    index = IndexList{{1, 1}};
    x = Point{{5., 7.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{10., 7.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{5., 10.}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
    x = Point{{3.51, 5.01}};
    assert(numerical::areEqual(f(x), data(index)));
    f.snapToGrid(x, &i);
    assert(i == index);
  }
  //
  // 2-D, linear interpolation, plain.
  //
  {
    const std::size_t Dimension = 2;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    {
      const SizeList extents = {{5, 7}};
      Grid grid(extents);
      const Point lower = {{0, 0}};
      const Point upper = {{2, 3}};
      const Point dx = (upper - lower) /
        ext::convert_array<double>(extents - std::size_t(1));
      const Iterator end = Iterator::end(grid.range());
      Point x;
      for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
        x = lower + stlib::ext::convert_array<double>(*i) * dx;
        grid(*i) = stlib::ext::sum(x);
      }
      const BBox domain = {lower, upper};
      InterpolatingFunction f(grid, domain);
      for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
        x = lower + stlib::ext::convert_array<double>(*i) * dx;
        assert(numerical::areEqual(f(x), stlib::ext::sum(x)));
      }
    }
    {
      // Make an array. Each element value is the sum of its index list.
      SizeList extents = {{2, 2}};
      Grid data(extents);
      const Iterator end = Iterator::end(data.range());
      for (Iterator i = Iterator::begin(data.range()); i != end; ++i) {
        data(*i) = stlib::ext::sum(*i);
      }
      BBox domain = {{{2., 3.}}, {{5., 7.}}};
      InterpolatingFunction f(data, domain);
      Point x;
      IndexList i, index;

      x = Point{{2., 3.}};
      index = IndexList{{0, 0}};
      assert(numerical::areEqual(f(x), data(index)));
      f.snapToGrid(x, &i);
      assert(i == index);

      x = Point{{2., 7.}};
      index = IndexList{{0, 1}};
      assert(numerical::areEqual(f(x), data(index)));
      f.snapToGrid(x, &i);
      assert(i == index);

      x = Point{{5., 3.}};
      index = IndexList{{1, 0}};
      assert(numerical::areEqual(f(x), data(index)));
      f.snapToGrid(x, &i);
      assert(i == index);

      x = Point{{5., 7.}};
      index = IndexList{{1, 1}};
      assert(numerical::areEqual(f(x), data(index)));
      f.snapToGrid(x, &i);
      assert(i == index);
    }
  }
  //
  // 2-D, linear interpolation, periodic.
  //
  {
    const std::size_t Dimension = 2;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    {
      // Define a grid to sample the function at every 30 degrees.
      const SizeList extents = {{12, 12}};
      Grid grid(extents);
      const Point lower = {{0, 0}};
      const Point upper = {{360, 360}};
      const Point dx = (upper - lower) / ext::convert_array<double>(extents);
      // Sample the function.
      const double Deg = numerical::Constants<double>::Degree();
      const Iterator end = Iterator::end(grid.range());
      Point x;
      for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
        x = lower + stlib::ext::convert_array<double>(*i) * dx;
        const double y = std::cos(x[0] * Deg) * std::sin(x[1] * Deg);
        grid(*i) = y;
      }
      // Construct an interpolating function.
      const BBox domain = {lower, upper};
      InterpolatingFunction f(grid, domain);
      // Ensure that the function has the correct values at the grid points.
      for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
        x = lower + stlib::ext::convert_array<double>(*i) * dx;
        const double y = std::cos(x[0] * Deg) * std::sin(x[1] * Deg);
        assert(numerical::areEqual(f(x), y));
      }
    }
    {
      SizeList extents = {{2, 2}};
      Grid data(extents);
      data.fill(0);
      data(IndexList{{1, 1}}) = 1;
      BBox domain = {{{2., 3.}}, {{5., 7.}}};
      InterpolatingFunction f(data, domain);
      assert(numerical::areEqual(f(Point{{2., 3.}}), 0.));
      assert(numerical::areEqual(f(Point{{2., 7.}}), 0.));
      assert(numerical::areEqual(f(Point{{5., 3.}}), 0.));
      assert(numerical::areEqual(f(Point{{5., 7.}}), 0.));
      assert(numerical::areEqual(f(Point{{3.5, 5.}}), 1.));
    }
  }
  //
  // 3-D, linear interpolation, plain.
  //
  {
    const std::size_t Dimension = 3;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    // Make an array. Each element value is the sum of its index list.
    SizeList extents = {{2, 2, 2}};
    Grid data(extents);
    const Iterator end = Iterator::end(data.range());
    for (Iterator i = Iterator::begin(data.range()); i != end; ++i) {
      data(*i) = stlib::ext::sum(*i);
    }
    BBox domain = {{{2., 3., 5.}}, {{7., 11., 13.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(Point{{2., 3., 5.}}),
                               data(IndexList{{0, 0, 0}})));
    assert(numerical::areEqual(f(Point{{7., 3., 5.}}),
                               data(IndexList{{1, 0, 0}})));
    assert(numerical::areEqual(f(Point{{2., 11., 5.}}),
                               data(IndexList{{0, 1, 0}})));
    assert(numerical::areEqual(f(Point{{7., 11., 5.}}),
                               data(IndexList{{1, 1, 0}})));
    assert(numerical::areEqual(f(Point{{2., 3., 13.}}),
                               data(IndexList{{0, 0, 1}})));
    assert(numerical::areEqual(f(Point{{7., 3., 13.}}),
                               data(IndexList{{1, 0, 1}})));
    assert(numerical::areEqual(f(Point{{2., 11., 13.}}),
                               data(IndexList{{0, 1, 1}})));
    assert(numerical::areEqual(f(Point{{7., 11., 13.}}),
                               data(IndexList{{1, 1, 1}})));
  }
  //
  // 3-D, linear interpolation, periodic.
  //
  {
    const std::size_t Dimension = 3;
    const std::size_t Order = 1;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::SizeList SizeList;

    SizeList extents = {{2, 2, 2}};
    Grid data(extents);
    data.fill(0);
    data(IndexList{{1, 1, 1}}) = 1;
    BBox domain = {{{2., 3., 5.}}, {{7., 11., 13.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(Point{{2., 3., 5.}}), 0.));
    assert(numerical::areEqual(f(Point{{7., 3., 5.}}), 0.));
    assert(numerical::areEqual(f(Point{{2., 11., 5.}}), 0.));
    assert(numerical::areEqual(f(Point{{7., 11., 5.}}), 0.));
    assert(numerical::areEqual(f(Point{{2., 3., 13.}}), 0.));
    assert(numerical::areEqual(f(Point{{7., 3., 13.}}), 0.));
    assert(numerical::areEqual(f(Point{{2., 11., 13.}}), 0.));
    assert(numerical::areEqual(f(Point{{7., 11., 13.}}), 0.));
    assert(numerical::areEqual(f(Point{{4.5, 7., 9.}}), 1.));
  }
  //
  // 1-D, cubic interpolation, plain.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 3;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;

    {
      SizeList extents = {{2}};
      Grid data(extents);
      data[0] = 2;
      data[1] = 3;
      BBox domain = {{{5.}}, {{7.}}};
      InterpolatingFunction f(data, domain);
      assert(numerical::areEqual(f(domain.lower), data[0]));
      assert(numerical::areEqual(f(domain.upper), data[1]));
      Point x;
      x[0] = 6;
      assert(numerical::areEqual(f(x), 2.5));
      std::array<double, 1> gradient;
      assert(numerical::areEqual(f(domain.lower, &gradient),
                                 data[0]));
      assert(numerical::areEqual(gradient[0], 0.5));
      assert(numerical::areEqual(f(domain.upper, &gradient),
                                 data[1]));
      assert(numerical::areEqual(gradient[0], 0.5));
      x[0] = 6;
      assert(numerical::areEqual(f(x, &gradient), 2.5));
      assert(numerical::areEqual(gradient[0], 0.5));
    }
    {
      // x^3.
      // For the center cell the interpolating function is
      // y = -6 + 13 x - 9 x^2 + 3 x^3.
      // y' = 13 - 18 x + 9 x^2
      SizeList extents = {{4}};
      Grid data(extents);
      data[0] = 0;
      data[1] = 1;
      data[2] = 8;
      data[3] = 27;
      BBox domain = {{{0.}}, {{3.}}};
      InterpolatingFunction f(data, domain);
      Point x;
      for (std::size_t i = 0; i != 4; ++i) {
        x[0] = i;
        assert(numerical::areEqual(f(x), data[i]));
      }
      x[0] = 1.5;
      assert(numerical::areEqual(f(x), -6 + 13 * x[0] - 9 * x[0] * x[0]
                                 + 3 * x[0] * x[0] * x[0]));
      std::array<double, 1> gradient;
      assert(numerical::areEqual(f(x, &gradient), -6 + 13 * x[0]
                                 - 9 * x[0] * x[0]
                                 + 3 * x[0] * x[0] * x[0]));
      assert(numerical::areEqual(gradient[0], 13 - 18 * x[0]
                                 + 9 * x[0] * x[0]));
    }
  }
  //
  // 1-D, cubic interpolation, periodic.
  //
  {
    const std::size_t Dimension = 1;
    const std::size_t Order = 3;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::SizeList SizeList;

    SizeList extents = {{1}};
    Grid data(extents);
    data[0] = 2;
    BBox domain = {{{5.}}, {{7.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(domain.lower), data[0]));
    assert(numerical::areEqual(f(domain.upper), data[0]));
  }
  //
  // 2-D, cubic interpolation, plain.
  //
  {
    const std::size_t Dimension = 2;
    const std::size_t Order = 3;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<Dimension> Iterator;

    // Make an array. Each element value is the sum of its index list.
    SizeList extents = {{2, 2}};
    Grid data(extents);
    const Iterator end = Iterator::end(data.range());
    for (Iterator i = Iterator::begin(data.range()); i != end; ++i) {
      data(*i) = stlib::ext::sum(*i);
    }
    BBox domain = {{{2., 3.}}, {{5., 7.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(Point{{2., 3.}}),
                               data(IndexList{{0, 0}})));
    assert(numerical::areEqual(f(Point{{2., 7.}}),
                               data(IndexList{{0, 1}})));
    assert(numerical::areEqual(f(Point{{5., 3.}}),
                               data(IndexList{{1, 0}})));
    assert(numerical::areEqual(f(Point{{5., 7.}}),
                               data(IndexList{{1, 1}})));
    std::array<double, 2> gradient;
    assert(numerical::areEqual(f(Point{{2., 3.}}, &gradient),
                               data(IndexList{{0, 0}})));
    assert(numerical::areEqual(gradient, Point{{1. / 3., 1. / 4.}}));
    assert(numerical::areEqual(f(Point{{2., 7.}}, &gradient),
                               data(IndexList{{0, 1}})));
    assert(numerical::areEqual(f(Point{{5., 3.}}, &gradient),
                               data(IndexList{{1, 0}})));
    assert(numerical::areEqual(gradient, Point{{1. / 3., 1. / 4.}}));
    assert(numerical::areEqual(f(Point{{5., 7.}}, &gradient),
                               data(IndexList{{1, 1}})));
    assert(numerical::areEqual(gradient, Point{{1. / 3., 1. / 4.}}));
  }
  //
  // 2-D, cubic interpolation, periodic.
  //
  {
    const std::size_t Dimension = 2;
    const std::size_t Order = 3;
    typedef numerical::InterpolatingFunctionRegularGrid < double, Dimension,
            Order, true > InterpolatingFunction;
    typedef InterpolatingFunction::Point Point;
    typedef InterpolatingFunction::BBox BBox;
    typedef InterpolatingFunction::Grid Grid;
    typedef Grid::IndexList IndexList;
    typedef Grid::SizeList SizeList;

    SizeList extents = {{2, 2}};
    Grid data(extents);
    data.fill(0);
    data(IndexList{{1, 1}}) = 1;
    BBox domain = {{{2., 3.}}, {{5., 7.}}};
    InterpolatingFunction f(data, domain);
    assert(numerical::areEqual(f(Point{{2., 3.}}), 0.));
    assert(numerical::areEqual(f(Point{{2., 7.}}), 0.));
    assert(numerical::areEqual(f(Point{{5., 3.}}), 0.));
    assert(numerical::areEqual(f(Point{{5., 7.}}), 0.));
    assert(numerical::areEqual(f(Point{{3.5, 5.}}), 1.));
  }

  return 0;
}

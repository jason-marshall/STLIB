// -*- C++ -*-

#include "stlib/numerical/interpolation/LinInterpGrid.h"

#include <iostream>
#include <limits>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

double
f(const std::array<double, 1>& x)
{
  return x[0];
}

double
f(const std::array<double, 2>& x)
{
  return x[0] + 2 * x[1];
}

double
f(const std::array<double, 3>& x)
{
  return x[0] + 2 * x[1] + 3 * x[2];
}

int
main()
{
  using namespace numerical;

  const double eps = 100 * std::numeric_limits<double>::epsilon();

  //
  // 1-D
  //
  {
    const std::size_t N = 1;
    typedef container::MultiArray<double, N> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<N> Iterator;
    typedef LinInterpGrid<N>::Point Point;

    container::MultiArray<double, N> a(ext::filled_array<SizeList>(10));
    for (Iterator i = Iterator::begin(a.range()); i != Iterator::end(a.range());
         ++i) {
      a(*i) = (*i)[0];
    }

    const geom::BBox<double, N>
    domain = {ext::filled_array<Point>(0),
              ext::convert_array<double>(a.extents()) - 1.
             };
    LinInterpGrid<N> x(a, domain);

    SizeList i;
    Point p;
    for (i[0] = 0; i[0] != a.extents()[0] - 1; ++i[0]) {
      p[0] = i[0] + 0.25;
      assert(std::abs(x(p) - f(p)) < eps);
    }
  }
  //
  // 2-D
  //
  {
    const std::size_t N = 2;
    typedef container::MultiArray<double, N> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<N> Iterator;
    typedef LinInterpGrid<N>::Point Point;

    container::MultiArray<double, N> a(SizeList{{10, 20}});
    for (Iterator i = Iterator::begin(a.range()); i != Iterator::end(a.range());
         ++i) {
      a(*i) = (*i)[0] + 2 * (*i)[1];
    }

    const geom::BBox<double, N> domain = {
      ext::filled_array<Point>(0),
      ext::convert_array<double>(a.extents()) - 1.
    };
    LinInterpGrid<N> x(a, domain);

    SizeList i;
    Point p;
    for (i[0] = 0; i[0] != a.extents()[0] - 1; ++i[0]) {
      for (i[1] = 0; i[1] != a.extents()[1] - 1; ++i[1]) {
        p[0] = i[0] + 0.25;
        p[1] = i[1] + 0.5;
        assert(std::abs(x(p) - f(p)) < eps);
      }
    }
  }
  //
  // 3-D
  //
  {
    const std::size_t N = 3;
    typedef container::MultiArray<double, N> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef container::MultiIndexRangeIterator<N> Iterator;
    typedef LinInterpGrid<N>::Point Point;

    container::MultiArray<double, N> a(SizeList{{10, 20, 30}});
    for (Iterator i = Iterator::begin(a.range()); i != Iterator::end(a.range());
         ++i) {
      a(*i) = (*i)[0] + 2 * (*i)[1] + 3 * (*i)[2];
    }

    const geom::BBox<double, N> domain = {
      ext::filled_array<Point>(0),
      ext::convert_array<double>(a.extents()) - 1.
    };
    LinInterpGrid<N> x(a, domain);

    SizeList i;
    Point p;
    for (i[0] = 0; i[0] != a.extents()[0] - 1; ++i[0]) {
      for (i[1] = 0; i[1] != a.extents()[1] - 1; ++i[1]) {
        for (i[2] = 0; i[2] != a.extents()[2] - 1; ++i[2]) {
          p[0] = i[0] + 0.25;
          p[1] = i[1] + 0.5;
          p[2] = i[2] + 0.75;
          assert(std::abs(x(p) - f(p)) < eps);
        }
      }
    }
  }
  return 0;
}

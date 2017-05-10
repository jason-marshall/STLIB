// -*- C++ -*-

#include "stlib/levelSet/contentFromDistance.h"

#include "stlib/numerical/equality.h"
#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexExtentsIterator.h"

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  using numerical::areEqual;
  using levelSet::contentFromDistance;

  typedef float T;

  // 1-D
  {
    const std::size_t D = 1;

    // Patch.
    const T Dx = 0.1;
    {
      const std::array<T, 4> data = {{1, 1, 1, 1}};
      assert(areEqual(contentFromDistance<D>(data.begin(), data.end(), Dx),
                      T(0)));
    }
    {
      const std::array<T, 4> data = {{ -1, -1, -1, -1}};
      assert(areEqual(contentFromDistance<D>(data.begin(), data.end(), Dx),
                      T(data.size() * Dx)));
    }
    {
      const std::array<T, 4> data = {{ -Dx, 0, Dx, 2 * Dx}};
      assert(areEqual(contentFromDistance<D>(data.begin(), data.end(), Dx),
                      T(1.5 * Dx)));
    }
  }

  // 2-D
  {
    const std::size_t D = 2;

    {
      const T Dx = 0.1;
      {
        const std::array<T, 1> data = {{1}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx), T(0)));
      }
      {
        const std::array<T, 1> data = {{ -1}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx), T(Dx * Dx)));
      }
      {
        const std::array<T, 1> data = {{0}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx), T(0.5 * Dx * Dx)));
      }
    }
    {
      typedef container::SimpleMultiArray<T, D> MultiArray;
      typedef MultiArray::IndexList IndexList;
      typedef container::SimpleMultiIndexExtentsIterator<D> Iterator;
      typedef std::array<T, D> Point;

      const T HalfRadius = 1.5;
      const Point Lower = {{ -HalfRadius, -HalfRadius}};
      const Point Upper = {{HalfRadius, HalfRadius}};
      MultiArray ball(ext::filled_array<IndexList>(10));
      const T Dx = (Upper[0] - Lower[0]) / (ball.extents()[0] - 1);

      Point x;
      const Iterator iEnd = Iterator::end(ball.extents());
      for (Iterator i = Iterator::begin(ball.extents()); i != iEnd; ++i) {
        x = Lower + stlib::ext::convert_array<T>(*i) * Dx;
        ball(*i) = stlib::ext::magnitude(x) - 1;
      }

      const T content = contentFromDistance<2>(ball.begin(), ball.end(), Dx);
      const T Exact = numerical::Constants<T>::Pi();
      const T error = std::abs((content - Exact) / Exact);
      std::cout << "Content of 2-D ball = " << content
                << ", Error = " << error << '\n';
      assert(error < 0.01);
    }
  }

  // 3-D
  {
    const std::size_t D = 3;

    {
      const T Dx = 0.1;
      {
        const std::array<T, 1> data = {{1}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx), T(0)));
      }
      {
        const std::array<T, 1> data = {{ -1}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx), T(Dx * Dx * Dx)));
      }
      {
        const std::array<T, 1> data = {{0}};
        assert(areEqual(contentFromDistance<D>
                        (data.begin(), data.end(), Dx),
                        T(0.5 * Dx * Dx * Dx)));
      }
    }
    {
      typedef container::SimpleMultiArray<T, D> MultiArray;
      typedef MultiArray::IndexList IndexList;
      typedef container::SimpleMultiIndexExtentsIterator<D> Iterator;
      typedef std::array<T, D> Point;

      const T HalfRadius = 1.5;
      const Point Lower = {{ -HalfRadius, -HalfRadius, -HalfRadius}};
      const Point Upper = {{HalfRadius, HalfRadius, HalfRadius}};
      MultiArray ball(ext::filled_array<IndexList>(10));
      const T Dx = (Upper[0] - Lower[0]) / (ball.extents()[0] - 1);

      Point x;
      const Iterator iEnd = Iterator::end(ball.extents());
      for (Iterator i = Iterator::begin(ball.extents()); i != iEnd; ++i) {
        x = Lower + stlib::ext::convert_array<T>(*i) * Dx;
        ball(*i) = stlib::ext::magnitude(x) - 1;
      }

      const T content = contentFromDistance<3>(ball.begin(), ball.end(), Dx);
      const T Exact = 4 * numerical::Constants<T>::Pi() / 3;
      const T error = std::abs((content - Exact) / Exact);
      std::cout << "Content of 3-D ball = " << content
                << ", Error = " << error << '\n';
      assert(error < 0.04);
    }
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/container/Tensor.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 1-D
  //
  {
    {
      container::Tensor1<double, 0> x;
      assert(x.empty());
    }
    {
      container::Tensor1<double, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      std::array<float, 1> data = {{7}};
      {
        container::Tensor1<double, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::Tensor1<double, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::Tensor1<double, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      container::Tensor1<double, 1> x;
      assert(x.extents() == (std::array<std::size_t, 1>{{1}}));
      x[0] = 7;
      assert(x(0) == x[0]);
      assert(x(std::array<std::size_t, 1>{{0}}) == x[0]);
      x(0) = 11;
      assert(x[0] == 11);
      x(std::array<std::size_t, 1>{{0}}) = 13;
      assert(x[0] == 13);
    }
  }
  //
  // 2-D
  //
  {
    {
      std::array<float, 1> data = {{7}};
      {
        container::Tensor2<double, 1, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::Tensor2<double, 1, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::Tensor2<double, 1, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      container::Tensor2<double, 1, 1> x;
      assert(x.extents() == (std::array<std::size_t, 2>{{1, 1}}));
      x[0] = 7;
      assert(x(0, 0) == x[0]);
      assert(x(std::array<std::size_t, 2>{{0, 0}}) == x[0]);
      x(0, 0) = 11;
      assert(x[0] == 11);
      x(std::array<std::size_t, 2>{{0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      const double data[] = {
        2, 3,
        5, 7,
        11, 13
      };
      container::Tensor2<double, 2, 3> x(data);
      assert(x.extents() == (std::array<std::size_t, 2>{{2, 3}}));
      std::size_t n = 0;
      for (std::size_t j = 0; j != 3; ++j) {
        for (std::size_t i = 0; i != 2; ++i) {
          assert(x(i, j) == data[n++]);
        }
      }
    }
  }
  //
  // 3-D
  //
  {
    {
      std::array<float, 1> data = {{7}};
      {
        container::Tensor3<double, 1, 1, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::Tensor3<double, 1, 1, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::Tensor3<double, 1, 1, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      container::Tensor3<double, 1, 1, 1> x;
      assert(x.extents() == (std::array<std::size_t, 3>{{1, 1, 1}}));
      x[0] = 7;
      assert(x(0, 0, 0) == x[0]);
      assert(x(std::array<std::size_t, 3>{{0, 0, 0}}) == x[0]);
      x(0, 0, 0) = 11;
      assert(x[0] == 11);
      x(std::array<std::size_t, 3>{{0, 0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      double data[2 * 3 * 5];
      for (std::size_t i = 0; i != 2 * 3 * 5; ++i) {
        data[i] = i;
      }
      container::Tensor3<double, 2, 3, 5> x(data);
      assert(x.extents() == (std::array<std::size_t, 3>{{2, 3, 5}}));
      std::size_t n = 0;
      for (std::size_t k = 0; k != 5; ++k) {
        for (std::size_t j = 0; j != 3; ++j) {
          for (std::size_t i = 0; i != 2; ++i) {
            assert(x(i, j, k) == data[n++]);
          }
        }
      }
    }
  }
  return 0;
}

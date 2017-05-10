// -*- C++ -*-

#include "stlib/container/EquilateralArray.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

template<std::size_t D>
void
test()
{
  std::array<float, 1> data = {{7}};
  container::EquilateralArray<double, D, 1> x(data);
  assert(x[0] == data[0]);
  const std::array<std::size_t, D> i = {{0}};
  assert(x(i) == data[0]);
}


int
main()
{
  //
  // 0-D
  //
  {
    {
      container::EquilateralArray<double, 0, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      std::array<float, 1> data = {{7}};
      {
        container::EquilateralArray<double, 0, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 0, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 0, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 0, 1> x(data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      const std::array<std::size_t, 0> index = {{}};
      container::EquilateralArray<double, 0, 1> x;
      assert(x.extents() == index);
      x[0] = 7;
      assert(x() == x[0]);
      assert(x(index) == x[0]);
      x() = 11;
      assert(x[0] == 11);
      x(index) = 13;
      assert(x[0] == 13);
    }
  }
  //
  // 1-D
  //
  {
    {
      container::EquilateralArray<double, 1, 0> x;
      assert(x.empty());
    }
    {
      container::EquilateralArray<double, 1, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      std::array<float, 1> data = {{7}};
      {
        container::EquilateralArray<double, 1, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 1, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 1, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 1, 1> x(data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArray<double, 1, 1> EquilateralArray;
      typedef EquilateralArray::IndexList IndexList;
      EquilateralArray x;
      assert(x.extents() == IndexList{{1}});
      x[0] = 7;
      assert(x(0) == x[0]);
      assert(x(IndexList{{0}}) == x[0]);
      x(0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0}}) = 13;
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
        container::EquilateralArray<double, 2, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 2, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 2, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 2, 1> x(data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArray<double, 2, 1> EquilateralArray;
      typedef EquilateralArray::IndexList IndexList;
      EquilateralArray x;
      assert(x.extents() == (IndexList{{1, 1}}));
      x[0] = 7;
      assert(x(0, 0) == x[0]);
      assert(x(IndexList{{0, 0}}) == x[0]);
      x(0, 0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      typedef container::EquilateralArray<double, 2, 2> EquilateralArray;
      typedef EquilateralArray::IndexList IndexList;
      const double data[] = {
        2, 3,
        5, 7
      };
      EquilateralArray x(data);
      assert(x.extents() == (IndexList{{2, 2}}));
      std::size_t n = 0;
      for (std::size_t j = 0; j != 2; ++j) {
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
        container::EquilateralArray<double, 3, 1> x(data);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 3, 1> x(data.begin(), data.end());
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 3, 1> x(&data[0]);
        assert(x[0] == data[0]);
      }
      {
        container::EquilateralArray<double, 3, 1> x(data[0]);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArray<double, 3, 1> EquilateralArray;
      typedef EquilateralArray::IndexList IndexList;
      EquilateralArray x;
      assert(x.extents() == (IndexList{{1, 1, 1}}));
      x[0] = 7;
      assert(x(0, 0, 0) == x[0]);
      assert(x(IndexList{{0, 0, 0}}) == x[0]);
      x(0, 0, 0) = 11;
      assert(x[0] == 11);
      x(IndexList{{0, 0, 0}}) = 13;
      assert(x[0] == 13);
    }
    {
      double data[2 * 2 * 2];
      for (std::size_t i = 0; i != 2 * 2 * 2; ++i) {
        data[i] = i;
      }
      container::EquilateralArray<double, 3, 2> x(data);
      assert((x.extents() ==
              ext::filled_array<std::array<std::size_t, 3> >(2)));
      std::size_t n = 0;
      for (std::size_t k = 0; k != 2; ++k) {
        for (std::size_t j = 0; j != 2; ++j) {
          for (std::size_t i = 0; i != 2; ++i) {
            assert(x(i, j, k) == data[n++]);
          }
        }
      }
    }
  }
  //
  // Higher dimensions.
  //
  test<4>();
  test<5>();
  test<6>();
  test<7>();
  test<8>();
  test<9>();
  test<10>();

  return 0;
}

// -*- C++ -*-

#include "stlib/container/EquilateralArrayRef.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

template<std::size_t D>
void
test()
{
  typename container::EquilateralArrayRef<double, D, 1>::value_type data[] = {7};
  container::EquilateralArrayRef<double, D, 1> x(data);
  assert(x[0] == data[0]);
  const std::array<std::size_t, D> i = {{0}};
  assert(x(i) == data[0]);
}


int
main()
{
  {
    typedef container::EquilateralArrayRef<double, 0, 1> T;
    {
      T::value_type x = 0;
      assert(x == 0);
      {
        T::reference y = x;
        assert(y == 0);
      }
      {
        T::const_reference y = x;
        assert(y == 0);
      }
    }
    {
      T::iterator x = 0;
      assert(x == 0);
    }
    {
      T::const_iterator x = 0;
      assert(x == 0);
    }
    {
      T::size_type x = 0;
      assert(x == 0);
    }
    {
      T::difference_type x = 0;
      assert(x == 0);
    }
    {
      T::reverse_iterator x;
    }
    {
      T::const_reverse_iterator x;
    }
  }
  //
  // 0-D
  //
  {
    {
      container::EquilateralArrayRef<double, 0, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      double data[] = {7};
      {
        container::EquilateralArrayRef<double, 0, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      double data[1];
      const std::array<std::size_t, 0> index = {{}};
      container::EquilateralArrayRef<double, 0, 1> x(data);
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
      container::EquilateralArrayRef<double, 1, 0> x;
      assert(x.empty());
    }
    {
      container::EquilateralArrayRef<double, 1, 1> x;
      assert(! x.empty());
      assert(x.size() == 1);
    }
    {
      double data[] = {7};
      {
        container::EquilateralArrayRef<double, 1, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArrayRef<double, 1, 1> EquilateralArrayRef;
      typedef EquilateralArrayRef::IndexList IndexList;
      double data[1];
      EquilateralArrayRef x(data);
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
      double data[] = {7};
      {
        container::EquilateralArrayRef<double, 2, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArrayRef<double, 2, 1> EquilateralArrayRef;
      typedef EquilateralArrayRef::IndexList IndexList;
      double data[1];
      EquilateralArrayRef x(data);
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
      typedef container::EquilateralArrayRef<double, 2, 2> EquilateralArrayRef;
      typedef EquilateralArrayRef::IndexList IndexList;
      double data[] = {
        2, 3,
        5, 7
      };
      EquilateralArrayRef x(data);
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
      double data[] = {7};
      {
        container::EquilateralArrayRef<double, 3, 1> x(data);
        assert(x[0] == data[0]);
      }
    }
    {
      typedef container::EquilateralArrayRef<double, 3, 1> EquilateralArrayRef;
      typedef EquilateralArrayRef::IndexList IndexList;
      double data[1];
      EquilateralArrayRef x(data);
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
      container::EquilateralArrayRef<double, 3, 2> x(data);
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

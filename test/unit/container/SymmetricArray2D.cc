// -*- C++ -*-

#include "stlib/container/SymmetricArray2D.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

template<typename _T>
bool
isValid(const container::SymmetricArray2D<_T, false>& x)
{
  return x.extent() * (x.extent() - 1) / 2 == x.size();
}

template<typename _T>
bool
isValid(const container::SymmetricArray2D<_T, true>& x)
{
  return (x.extent() + 1) * x.extent() / 2 == x.size();
}

int
main()
{
  // Default constructor.
  {
    container::SymmetricArray2D<double, false> x;
    assert(isValid(x));
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  {
    container::SymmetricArray2D<double, true> x;
    assert(isValid(x));
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  // Size constructor. Extent == 0.
  {
    const std::size_t Extent = 0;
    {
      container::SymmetricArray2D<double, false> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::SymmetricArray2D<double, true> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
  }
  // Size constructor. Extent == 1.
  {
    const std::size_t Extent = 1;
    {
      container::SymmetricArray2D<double, false> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::SymmetricArray2D<double, true> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), 0.)) == x.size());
      x[0] = 1;
      assert(x[0] == 1);
      x(0, 0) = 2;
      assert(x(0, 0) == 2);
    }
  }
  // Size constructor. Extent == 2.
  {
    const std::size_t Extent = 2;
    {
      container::SymmetricArray2D<double, false> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), 0.)) == x.size());
      x[0] = 1;
      assert(x[0] == 1);
      x(0, 1) = 2;
      assert(x(0, 1) == 2);
    }
    {
      container::SymmetricArray2D<double, true> x(Extent);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 3);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), 0.)) == x.size());
      x[0] = 1;
      x[1] = 1;
      x[2] = 1;
      assert(x[0] == 1 && x[1] == 1 && x[2] == 1);
      x(0, 0) = 2;
      x(0, 1) = 2;
      x(1, 1) = 2;
      assert(x(0, 0) == 2 && x(0, 1) == 2 && x(1, 1) == 2);
    }
  }
  // Size and initial value constructor. Extent == 2.
  {
    const std::size_t Extent = 2;
    const double InitialValue = 7;
    {
      container::SymmetricArray2D<double, false> x(Extent, InitialValue);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), InitialValue)) ==
             x.size());
      assert(x(0, 1) == InitialValue);
    }
    {
      container::SymmetricArray2D<double, true> x(Extent, InitialValue);
      assert(isValid(x));
      assert(x.extent() == Extent);
      assert(x.size() == 3);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), InitialValue)) ==
             x.size());
      assert(x(0, 0) == InitialValue && x(0, 1) == InitialValue &&
             x(1, 1) == InitialValue);
    }
  }
  {
    typedef container::SymmetricArray2D<double, true> SymmetricArray2D;
    SymmetricArray2D x(2);
    x(0, 0) = 2;
    x(0, 1) = 3;
    x(1, 1) = 5;

    // Copy constructor.
    {
      SymmetricArray2D y(x);
      assert(x == y);
    }

    // Swap.
    {
      SymmetricArray2D y(x);
      SymmetricArray2D z;
      z.swap(y);
      assert(x == z);
    }

    // Assignment operator
    {
      SymmetricArray2D y;
      y = x;
      assert(x == y);
    }

    // Ascii I/O.
    {
      std::stringstream s;
      s << x;
      SymmetricArray2D y;
      s >> y;
      assert(x == y);
    }
  }
  return 0;
}

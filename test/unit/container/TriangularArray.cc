// -*- C++ -*-

#include "stlib/container/TriangularArray.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  // Default constructor.
  {
    container::TriangularArray < double, container::LowerTriangular,
              container::StrictlyTriangular > x;
    assert(x.isValid());
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  {
    container::TriangularArray < double, container::LowerTriangular,
              container::NonStrictlyTriangular > x;
    assert(x.isValid());
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  {
    container::TriangularArray < double, container::UpperTriangular,
              container::StrictlyTriangular > x;
    assert(x.isValid());
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  {
    container::TriangularArray < double, container::UpperTriangular,
              container::NonStrictlyTriangular > x;
    assert(x.isValid());
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
      container::TriangularArray < double, container::LowerTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::TriangularArray < double, container::LowerTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
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
      container::TriangularArray < double, container::LowerTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::TriangularArray < double, container::LowerTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
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
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.begin() == x.end());
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
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
      container::TriangularArray < double, container::LowerTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), 0.)) == x.size());
      x[0] = 1;
      assert(x[0] == 1);
      x(1, 0) = 2;
      assert(x(1, 0) == 2);
    }
    {
      container::TriangularArray < double, container::LowerTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
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
      x(1, 0) = 2;
      x(1, 1) = 2;
      assert(x(0, 0) == 2 && x(1, 0) == 2 && x(1, 1) == 2);
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::StrictlyTriangular > x(Extent);
      assert(x.isValid());
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
      container::TriangularArray < double, container::UpperTriangular,
                container::NonStrictlyTriangular > x(Extent);
      assert(x.isValid());
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
      container::TriangularArray < double, container::LowerTriangular,
                container::StrictlyTriangular > x(Extent, InitialValue);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), InitialValue)) ==
             x.size());
      assert(x(1, 0) == InitialValue);
    }
    {
      container::TriangularArray < double, container::LowerTriangular,
                container::NonStrictlyTriangular > x(Extent, InitialValue);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 3);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), InitialValue)) ==
             x.size());
      assert(x(0, 0) == InitialValue && x(1, 0) == InitialValue &&
             x(1, 1) == InitialValue);
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::StrictlyTriangular > x(Extent, InitialValue);
      assert(x.isValid());
      assert(x.extent() == Extent);
      assert(x.size() == 1);
      assert(! x.empty());
      assert(x.begin() != x.end());
      assert(std::size_t(std::count(x.begin(), x.end(), InitialValue)) ==
             x.size());
      assert(x(0, 1) == InitialValue);
    }
    {
      container::TriangularArray < double, container::UpperTriangular,
                container::NonStrictlyTriangular > x(Extent, InitialValue);
      assert(x.isValid());
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
    typedef container::TriangularArray < double, container::UpperTriangular,
            container::NonStrictlyTriangular > TriangularArray;
    TriangularArray x(2);
    x(0, 0) = 2;
    x(0, 1) = 3;
    x(1, 1) = 5;

    // Copy constructor.
    {
      TriangularArray y(x);
      assert(x == y);
    }

    // Swap.
    {
      TriangularArray y(x);
      TriangularArray z;
      z.swap(y);
      assert(x == z);
    }

    // Assignment operator
    {
      TriangularArray y;
      y = x;
      assert(x == y);
    }

    // Ascii I/O.
    {
      std::stringstream s;
      s << x;
      TriangularArray y;
      s >> y;
      assert(x == y);
    }
  }
  return 0;
}

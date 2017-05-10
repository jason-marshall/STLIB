// -*- C++ -*-

//
// Tests for FixedArray<1,T>.
//

#include "stlib/ads/array/FixedArray.h"
#include "stlib/numerical/equality.h"

#include <iostream>
#include <sstream>
#include <vector>

using namespace stlib;

int
main()
{
  using namespace ads;
  using numerical::areEqual;

  {
    // sizeof
    static_assert(sizeof(FixedArray<1>) == sizeof(double), "sizeof check.");
  }
  {
    std::cout << "FixedArray<1>() = "
              << FixedArray<1>() << '\n';
    // Maximum size.
    std::cout << "max_size = " << FixedArray<1>::max_size() << '\n';
  }
  {
    typedef FixedArray<1> V;
    typedef FixedArray<1, V> A;
    const V a;
    A x(a);
  }
  {
    // Copy.
    double a[1] = {1};
    FixedArray<1> b;
    b.copy(a, a + 1);
    assert((b == FixedArray<1>(1)));
  }
  {
    // Swap.
    FixedArray<1> a(0.);
    FixedArray<1> ac(a);
    FixedArray<1> b(1.);
    FixedArray<1> bc(b);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    FixedArray<1> p(1.0);

    // accessors
    assert(p[0] == 1);
    assert(p[0] == 1);

    // manipulators
    p[0] = 2;
    assert(p[0] == 2);
    p[0] = 3;
    assert(p[0] == 3);
  }
  {
    // size()
    FixedArray< 1, std::vector<int> > a;
    assert(a.size() == 1);
    assert(! a.empty());
  }
  {
    // ==
    FixedArray<1, int> a(1);
    FixedArray<1, int> b(1);
    assert(a == b);
  }
  {
    // !=
    FixedArray<1, int> a(1);
    FixedArray<1, int> b(2);
    assert(a != b);
  }
  {
    // <

    // 1-D
    assert((FixedArray<1, int>(1)) < (FixedArray<1, int>(2)));
    assert((FixedArray<1, int>(2)) > (FixedArray<1, int>(1)));
    assert((FixedArray<1, int>(1)) <= (FixedArray<1, int>(2)));
    assert((FixedArray<1, int>(2)) >= (FixedArray<1, int>(1)));
    assert((FixedArray<1, int>(1)) <= (FixedArray<1, int>(1)));
    assert((FixedArray<1, int>(1)) >= (FixedArray<1, int>(1)));
  }
  {
    // +=
    FixedArray<1> a(1.), b(3.);
    FixedArray<1> v(2.);
    a += v;
    assert(a == b);
  }
  {
    // -=
    FixedArray<1> a(1.), b(-1.);
    FixedArray<1> v(2.);
    a -= v;
    assert(a == b);
  }
  //
  // Math operators.
  //
  {
    // sum
    FixedArray<1> x;
    x = 42;
    assert(computeSum(x) == 42);
  }
  {
    // product
    FixedArray<1> x;
    x = 42;
    assert(computeProduct(x) == 42);
  }
  {
    // min
    FixedArray<1> x;
    x = 42;
    assert(computeMinimum(x) == 42);
  }
  {
    // max
    FixedArray<1> x;
    x = 42;
    assert(computeMaximum(x) == 42);
  }
  {
    // negate
    FixedArray<1> a, ma;
    a = 7;
    ma = -7;
    a.negate();
    assert(a == ma);
  }
  {
    // sort
    {
      FixedArray<1> a(1);
      a.sort();
      assert(a.is_sorted());
    }
  }
  {
    // min_index
    {
      FixedArray<1> a(1);
      assert(a.min_index() == 0);
    }
  }
  {
    // max_index
    {
      FixedArray<1> a(1);
      assert(a.max_index() == 0);
    }
  }
  {
    // I/O
    std::stringstream file;
    file << FixedArray<1>(1.) << '\n';
    FixedArray<1> x;
    file >> x;
    assert((x == FixedArray<1>(1.)));
  }

  //
  // Standard math functions.
  //

  {
    typedef FixedArray<1> FA;
    FA a, b;

    a = abs(FA(-1.0));
    b = FA(std::abs(-1.0));
    assert(areEqual(a[0], b[0]));

    a = acos(FA(0.5));
    b = FA(std::acos(0.5));
    assert(areEqual(a[0], b[0]));

    a = asin(FA(0.5));
    b = FA(std::asin(0.5));
    assert(areEqual(a[0], b[0]));

    a = atan(FA(0.5));
    b = FA(std::atan(0.5));
    assert(areEqual(a[0], b[0]));

    a = ceil(FA(0.5));
    b = FA(std::ceil(0.5));
    assert(areEqual(a[0], b[0]));

    a = cos(FA(0.5));
    b = FA(std::cos(0.5));
    assert(areEqual(a[0], b[0]));

    a = cosh(FA(0.5));
    b = FA(std::cosh(0.5));
    assert(areEqual(a[0], b[0]));

    a = exp(FA(0.5));
    b = FA(std::exp(0.5));
    assert(areEqual(a[0], b[0]));

    a = floor(FA(0.5));
    b = FA(std::floor(0.5));
    assert(areEqual(a[0], b[0]));

    a = log(FA(0.5));
    b = FA(std::log(0.5));
    assert(areEqual(a[0], b[0]));

    a = log10(FA(0.5));
    b = FA(std::log10(0.5));
    assert(areEqual(a[0], b[0]));

    a = sin(FA(0.5));
    b = FA(std::sin(0.5));
    assert(areEqual(a[0], b[0]));

    a = sinh(FA(0.5));
    b = FA(std::sinh(0.5));
    assert(areEqual(a[0], b[0]));

    a = sqrt(FA(0.5));
    b = FA(std::sqrt(0.5));
    assert(areEqual(a[0], b[0]));

    a = tan(FA(0.5));
    b = FA(std::tan(0.5));
    assert(areEqual(a[0], b[0]));

    a = tanh(FA(0.5));
    b = FA(std::tanh(0.5));
    assert(areEqual(a[0], b[0]));
  }

  return 0;
}

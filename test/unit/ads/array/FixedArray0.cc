// -*- C++ -*-

//
// Tests for FixedArray<0,T>.
//

#include "stlib/ads/array/FixedArray.h"

#include <iostream>
#include <sstream>
#include <vector>

using namespace stlib;

int
main()
{
  using namespace ads;
  {
    // sizeof
    static_assert(sizeof(FixedArray<0>) == sizeof(char), "sizeof check.");
  }
  {
    std::cout << "FixedArray<0>() = "
              << FixedArray<0>() << '\n';
    // Maximum size.
    std::cout << "max_size = " << FixedArray<0>::max_size() << '\n';
  }
  {
    // Copy.
    double a[1] = {1};
    FixedArray<0> b;
    b.copy(a, a);
    assert((b == FixedArray<0>()));
  }
  {
    // Swap.
    FixedArray<0> a(0.);
    FixedArray<0> ac(a);
    FixedArray<0> b(1.);
    FixedArray<0> bc(b);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    // size()
    FixedArray< 0, std::vector<int> > a;
    assert(a.size() == 0);
    assert(a.empty());
  }
  {
    // ==
    FixedArray<0, int> a;
    FixedArray<0, int> b;
    assert(a == b);
  }
  {
    // +=
    FixedArray<0> a, b;
    FixedArray<0> v;
    a += v;
    assert(a == b);
  }
  {
    // -=
    FixedArray<0> a, b;
    FixedArray<0> v;
    a -= v;
    assert(a == b);
  }
  //
  // Math operators.
  //
  {
    // sum
    FixedArray<0> x;
    assert(computeSum(x) == 0);
  }
  {
    // product
    FixedArray<0> x;
    assert(computeProduct(x) == 1);
  }
  {
    // negate
    FixedArray<0> a, ma;
    a.negate();
    assert(a == ma);
  }
  {
    // sort
    FixedArray<0> a;
    a.sort();
    assert(a.is_sorted());
  }
  {
    // I/O
    std::stringstream file;
    file << FixedArray<0>() << '\n';
    FixedArray<0> x;
    file >> x;
    assert((x == FixedArray<0>()));
  }

  //
  // Standard math functions.
  //
  // N = 0.
  {
    FixedArray<0> x;
    assert(abs(x) == x);
    assert(acos(x) == x);
    assert(asin(x) == x);
    assert(atan(x) == x);
    assert(ceil(x) == x);
    assert(cos(x) == x);
    assert(cosh(x) == x);
    assert(exp(x) == x);
    assert(floor(x) == x);
    assert(log(x) == x);
    assert(log10(x) == x);
    assert(sin(x) == x);
    assert(sinh(x) == x);
    assert(sqrt(x) == x);
    assert(tan(x) == x);
    assert(tanh(x) == x);
  }

  return 0;
}

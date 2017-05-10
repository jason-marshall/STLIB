// -*- C++ -*-

//
// Tests for FixedArray<2,T>.
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
  using numerical::areSequencesEqual;

  {
    // sizeof
    static_assert(sizeof(FixedArray<2>) == 2 * sizeof(double), "sizeof check.");
  }
  {
    // 2 number constructor
    std::cout << "FixedArray<2>(1,2) = "
              << FixedArray<2>(1, 2) << '\n';
    // Maximum size.
    std::cout << "max_size = " << FixedArray<2>::max_size() << '\n';
  }
  {
    typedef FixedArray<1> V;
    typedef FixedArray<2, V> A;
    const V a;
    const V b;
    A x(a);
    A y(a, b);
  }
  {
    // Copy.
    double a[2] = {1, 2};
    FixedArray<2> b;
    b.copy(a, a + 2);
    assert((b == FixedArray<2>(1, 2)));
  }
  {
    // Swap.
    FixedArray<2> a(0, 1);
    FixedArray<2> ac(a);
    FixedArray<2> b(1, 2);
    FixedArray<2> bc(b);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    FixedArray<2> p(1, 2);

    // accessors
    assert(p[0] == 1 && p[1] == 2);
    assert(p[0] == 1 && p[1] == 2);

    // manipulators
    p[0] = 2;
    p[1] = 3;
    assert(p[0] == 2 && p[1] == 3);
    p[0] = 3;
    p[1] = 1;
    assert(p[0] == 3 && p[1] == 1);
  }
  {
    // size()
    FixedArray< 2, std::vector<int> > a;
    assert(a.size() == 2);
    assert(! a.empty());
  }
  {
    // ==
    FixedArray<2, int> a(1, 2);
    FixedArray<2, int> b(1, 2);
    assert(a == b);
  }
  {
    // !=
    FixedArray<2, int> a(1, 2);
    FixedArray<2, int> b(2, 3);
    assert(a != b);
  }
  {
    // <
    assert((FixedArray<2, int>(1, 2)) < (FixedArray<2, int>(2, 2)));
    assert((FixedArray<2, int>(1, 2)) < (FixedArray<2, int>(1, 3)));

    assert((FixedArray<2, int>(2, 2)) > (FixedArray<2, int>(1, 2)));
    assert((FixedArray<2, int>(1, 3)) > (FixedArray<2, int>(1, 2)));

    assert((FixedArray<2, int>(1, 2)) <= (FixedArray<2, int>(2, 2)));
    assert((FixedArray<2, int>(1, 2)) <= (FixedArray<2, int>(1, 3)));
    assert((FixedArray<2, int>(1, 2)) <= (FixedArray<2, int>(1, 2)));

    assert((FixedArray<2, int>(2, 2)) >= (FixedArray<2, int>(1, 2)));
    assert((FixedArray<2, int>(1, 3)) >= (FixedArray<2, int>(1, 2)));
    assert((FixedArray<2, int>(1, 2)) >= (FixedArray<2, int>(1, 2)));
  }
  {
    // +=
    FixedArray<2> a(1, 2), b(3, 5);
    FixedArray<2> v(2, 3);
    a += v;
    assert(a == b);
  }
  {
    // -=
    FixedArray<2> a(1, 2), b(-1, -1);
    FixedArray<2> v(2, 3);
    a -= v;
    assert(a == b);
  }
  //
  // Math operators.
  //
  {
    // sum
    assert(computeSum(FixedArray<2>(1, 2)) == 3);
  }
  {
    // product
    assert(computeProduct(FixedArray<2>(1, 2)) == 2);
  }
  {
    // min
    assert(computeMinimum(FixedArray<2>(1, 2)) == 1);
  }
  {
    // max
    assert(computeMaximum(FixedArray<2>(1, 2)) == 2);
  }
  {
    // negate
    FixedArray<2> b(1, 2);
    b.negate();
    assert((b == FixedArray<2>(-1, -2)));
  }
  {
    // sort
    {
      FixedArray<2> a(2, 1);
      assert(a.is_sorted(std::greater<double>()));
      a.sort();
      assert(a.is_sorted());
    }
  }
  {
    // min_index
    {
      FixedArray<2> a(1, 2);
      assert(a.min_index() == 0);
    }
    {
      FixedArray<2> a(2, 1);
      assert(a.min_index() == 1);
    }
  }
  {
    // max_index
    {
      FixedArray<2> a(1, 2);
      assert(a.max_index() == 1);
    }
    {
      FixedArray<2> a(2, 1);
      assert(a.max_index() == 0);
    }
  }
  {
    // I/O
    std::stringstream file;
    file << FixedArray<2>(1, 2) << '\n';
    FixedArray<2> x;
    file >> x;
    assert((x == FixedArray<2>(1, 2)));
  }

  //
  // Standard math functions.
  //
  {
    typedef FixedArray<2> FA;
    FA a, b;

    a = abs(FA(-1.0, 0.6));
    b = FA(std::abs(-1.0), std::abs(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = acos(FA(0.5, 0.6));
    b = FA(std::acos(0.5), std::acos(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = asin(FA(0.5, 0.6));
    b = FA(std::asin(0.5), std::asin(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = atan(FA(0.5, 0.6));
    b = FA(std::atan(0.5), std::atan(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = ceil(FA(0.5, 0.6));
    b = FA(std::ceil(0.5), std::ceil(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cos(FA(0.5, 0.6));
    b = FA(std::cos(0.5), std::cos(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cosh(FA(0.5, 0.6));
    b = FA(std::cosh(0.5), std::cosh(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = exp(FA(0.5, 0.6));
    b = FA(std::exp(0.5), std::exp(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = floor(FA(0.5, 0.6));
    b = FA(std::floor(0.5), std::floor(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log(FA(0.5, 0.6));
    b = FA(std::log(0.5), std::log(0.6));
    // CONTINUE. With GCC 4.7 the evaluation of log() at compile time is
    // wrong.
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log10(FA(0.5, 0.6));
    b = FA(std::log10(0.5), std::log10(0.6));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sin(FA(0.5, 0.6));
    b = FA(std::sin(0.5), std::sin(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sinh(FA(0.5, 0.6));
    b = FA(std::sinh(0.5), std::sinh(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sqrt(FA(0.5, 0.6));
    b = FA(std::sqrt(0.5), std::sqrt(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tan(FA(0.5, 0.6));
    b = FA(std::tan(0.5), std::tan(0.6));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tanh(FA(0.5, 0.6));
    b = FA(std::tanh(0.5), std::tanh(0.6));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));
  }

  return 0;
}

// -*- C++ -*-

//
// Tests for FixedArray<3,T>.
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
    static_assert(sizeof(FixedArray<3>) == 3 * sizeof(double),
                  "sizeof check.");
  }
  {
    // 3 number constructor
    std::cout << "FixedArray<3>(1,2,3) = "
              << FixedArray<3>(1, 2, 3) << '\n';
    // Maximum size.
    std::cout << "max_size = " << FixedArray<3>::max_size() << '\n';
  }
  {
    typedef FixedArray<2> V;
    typedef FixedArray<3, V> A;
    const V a;
    const V b;
    const V c;
    A x(a);
    A y(a, b, c);
  }
  {
    // Copy.
    double a[3] = {1, 2, 3};
    FixedArray<3> b;
    b.copy(a, a + 3);
    assert((b == FixedArray<3>(1, 2, 3)));
  }
  {
    // Swap.
    FixedArray<3> a(0, 1, 2);
    FixedArray<3> ac(a);
    FixedArray<3> b(1, 2, 3);
    FixedArray<3> bc(b);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    FixedArray<3> p(1, 2, 3);

    // accessors
    assert(p[0] == 1 && p[1] == 2 && p[2] == 3);
    assert(p[0] == 1 && p[1] == 2 && p[2] == 3);

    // manipulators
    p[0] = 2;
    p[1] = 3;
    p[2] = 5;
    assert(p[0] == 2 && p[1] == 3 && p[2] == 5);
    p[0] = 3;
    p[1] = 1;
    p[2] = 4;
    assert(p[0] == 3 && p[1] == 1 && p[2] == 4);
  }
  {
    // size()
    FixedArray< 4, std::vector<int> > a;
    assert(a.size() == 4);
    assert(! a.empty());
    assert(FixedArray<0>::empty());
  }
  {
    // ==
    FixedArray<3, int> a(1, 2, 3);
    FixedArray<3, int> b(1, 2, 3);
    assert(a == b);
  }
  {
    // !=
    FixedArray<3, int> a(1, 2, 3);
    FixedArray<3, int> b(2, 3, 5);
    assert(a != b);
  }
  {
    // <
    assert((FixedArray<3, int>(1, 2, 3)) < (FixedArray<3, int>(2, 2, 3)));
    assert((FixedArray<3, int>(1, 2, 3)) < (FixedArray<3, int>(1, 3, 3)));
    assert((FixedArray<3, int>(1, 2, 3)) < (FixedArray<3, int>(1, 2, 4)));

    assert((FixedArray<3, int>(2, 2, 3)) > (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(1, 3, 3)) > (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(1, 2, 4)) > (FixedArray<3, int>(1, 2, 3)));

    assert((FixedArray<3, int>(1, 2, 3)) <= (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(1, 2, 3)) <= (FixedArray<3, int>(2, 2, 3)));
    assert((FixedArray<3, int>(1, 2, 3)) <= (FixedArray<3, int>(1, 3, 3)));
    assert((FixedArray<3, int>(1, 2, 3)) <= (FixedArray<3, int>(1, 2, 4)));

    assert((FixedArray<3, int>(1, 2, 3)) >= (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(2, 2, 3)) >= (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(1, 3, 3)) >= (FixedArray<3, int>(1, 2, 3)));
    assert((FixedArray<3, int>(1, 2, 4)) >= (FixedArray<3, int>(1, 2, 3)));
  }
  {
    // +=
    FixedArray<3> a(1, 2, 3), b(3, 5, 8);
    FixedArray<3> v(2, 3, 5);
    a += v;
    assert(a == b);
  }
  {
    // -=
    FixedArray<3> a(1, 2, 3), b(-1, -1, -2);
    FixedArray<3> v(2, 3, 5);
    a -= v;
    assert(a == b);
  }
  //
  // Math operators.
  //
  {
    // sum
    assert(computeSum(FixedArray<3>(1, 2, 3)) == 6);
  }
  {
    // product
    assert(computeProduct(FixedArray<3>(1, 2, 3)) == 6);
  }
  {
    // min
    assert(computeMinimum(FixedArray<3>(1, 2, 3)) == 1);
    assert(computeMinimum(FixedArray<3>(3, 2, 1)) == 1);
  }
  {
    // max
    assert(computeMaximum(FixedArray<3>(1, 2, 3)) == 3);
    assert(computeMaximum(FixedArray<3>(3, 2, 1)) == 3);
  }
  {
    // negate
    FixedArray<3> c(1, 2, 3);
    c.negate();
    assert((c == FixedArray<3>(-1, -2, -3)));
  }
  {
    // sort
    FixedArray<3> a(3, 2, 1);
    assert(a.is_sorted(std::greater<double>()));
    a.sort();
    assert(a.is_sorted());
  }
  {
    // min_index
    {
      FixedArray<3> a(1, 2, 3);
      assert(a.min_index() == 0);
    }
    {
      FixedArray<3> a(1, 3, 2);
      assert(a.min_index() == 0);
    }
    {
      FixedArray<3> a(2, 1, 3);
      assert(a.min_index() == 1);
    }
    {
      FixedArray<3> a(2, 3, 1);
      assert(a.min_index() == 2);
    }
    {
      FixedArray<3> a(3, 1, 2);
      assert(a.min_index() == 1);
    }
    {
      FixedArray<3> a(3, 2, 1);
      assert(a.min_index() == 2);
    }
  }
  {
    // max_index
    {
      FixedArray<3> a(1, 2, 3);
      assert(a.max_index() == 2);
    }
    {
      FixedArray<3> a(1, 3, 2);
      assert(a.max_index() == 1);
    }
    {
      FixedArray<3> a(2, 1, 3);
      assert(a.max_index() == 2);
    }
    {
      FixedArray<3> a(2, 3, 1);
      assert(a.max_index() == 1);
    }
    {
      FixedArray<3> a(3, 1, 2);
      assert(a.max_index() == 0);
    }
    {
      FixedArray<3> a(3, 2, 1);
      assert(a.max_index() == 0);
    }
  }
  {
    // I/O
    std::stringstream file;
    file << FixedArray<3>(1, 2, 3) << '\n';
    FixedArray<3> x;
    file >> x;
    assert((x == FixedArray<3>(1, 2, 3)));
  }

  //
  // Standard math functions.
  //
  {
    typedef FixedArray<3> FA;
    FA a, b;

    a = abs(FA(-1.0, 0.6, 0.7));
    b = FA(std::abs(-1.0), std::abs(0.6), std::abs(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = acos(FA(0.5, 0.6, 0.7));
    b = FA(std::acos(0.5), std::acos(0.6), std::acos(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = asin(FA(0.5, 0.6, 0.7));
    b = FA(std::asin(0.5), std::asin(0.6), std::asin(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = atan(FA(0.5, 0.6, 0.7));
    b = FA(std::atan(0.5), std::atan(0.6), std::atan(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = ceil(FA(0.5, 0.6, 0.7));
    b = FA(std::ceil(0.5), std::ceil(0.6), std::ceil(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cos(FA(0.5, 0.6, 0.7));
    b = FA(std::cos(0.5), std::cos(0.6), std::cos(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cosh(FA(0.5, 0.6, 0.7));
    b = FA(std::cosh(0.5), std::cosh(0.6), std::cosh(0.7));
    // CONTINUE. With GCC 4.7 the evaluation of cosh() at compile time is
    // wrong.
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = exp(FA(0.5, 0.6, 0.7));
    b = FA(std::exp(0.5), std::exp(0.6), std::exp(0.7));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = floor(FA(0.5, 0.6, 0.7));
    b = FA(std::floor(0.5), std::floor(0.6), std::floor(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log(FA(0.5, 0.6, 0.7));
    b = FA(std::log(0.5), std::log(0.6), std::log(0.7));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log10(FA(0.5, 0.6, 0.7));
    b = FA(std::log10(0.5), std::log10(0.6), std::log10(0.7));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sin(FA(0.5, 0.6, 0.7));
    b = FA(std::sin(0.5), std::sin(0.6), std::sin(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sinh(FA(0.5, 0.6, 0.7));
    b = FA(std::sinh(0.5), std::sinh(0.6), std::sinh(0.7));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sqrt(FA(0.5, 0.6, 0.7));
    b = FA(std::sqrt(0.5), std::sqrt(0.6), std::sqrt(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tan(FA(0.5, 0.6, 0.7));
    b = FA(std::tan(0.5), std::tan(0.6), std::tan(0.7));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tanh(FA(0.5, 0.6, 0.7));
    b = FA(std::tanh(0.5), std::tanh(0.6), std::tanh(0.7));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));
  }

  return 0;
}

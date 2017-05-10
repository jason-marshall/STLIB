// -*- C++ -*-

//
// Tests for FixedArray<N,T>.
//

#include "stlib/ads/array/FixedArray.h"
#include "stlib/numerical/equality.h"

#include <iostream>
#include <sstream>

using namespace stlib;

int
main()
{
  using namespace ads;
  using numerical::areSequencesEqual;

  {
    // sizeof
    static_assert(sizeof(FixedArray<4>) == 4 * sizeof(double),
                  "sizeof check.");
  }
  {
    // default constructor
    std::cout << "FixedArray<5>() = "
              << FixedArray<5>() << '\n';
    FixedArray< 7, std::vector<int> > x;
    // Maximum size.
    std::cout << "max_size = " << FixedArray<0>::max_size() << '\n';
  }
  {
    typedef FixedArray<3> V;
    typedef FixedArray<4, V> A;
    const V a;
    const V b;
    const V c;
    const V d;
    A x(a);
    A y(a, b, c, d);
  }
  {
    // Copy.
    double a[4] = {1, 2, 3, 4};
    FixedArray<4> b;
    b.copy(a, a + 4);
    assert((b == FixedArray<4>(1, 2, 3, 4)));
  }
  {
    // Swap.
    FixedArray<4> a(0, 1, 2, 3);
    FixedArray<4> ac(a);
    FixedArray<4> b(1, 2, 3, 4);
    FixedArray<4> bc(b);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    FixedArray<4> p(1, 2, 3, 4);

    // accessors
    assert(p[0] == 1 && p[1] == 2 && p[2] == 3 && p[3] == 4);
    assert(p(0) == 1 && p(1) == 2 && p(2) == 3 && p(3) == 4);

    // manipulators
    p[0] = 2;
    p[1] = 3;
    p[2] = 5;
    p[3] = 7;
    assert(p[0] == 2 && p[1] == 3 && p[2] == 5 && p[3] == 7);
    p[0] = 3;
    p[1] = 1;
    p[2] = 4;
    p[3] = 1;
    assert(p[0] == 3 && p[1] == 1 && p[2] == 4 && p[3] == 1);
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
    FixedArray<4, int> a(1, 2, 3, 4);
    FixedArray<4, int> b(1, 2, 3, 4);
    assert(a == b);
  }
  {
    // !=
    FixedArray<4, int> a(1, 2, 3, 4);
    FixedArray<4, int> b(2, 3, 5, 7);
    assert(a != b);
  }
  {
    // <
    assert((FixedArray<4, int>(1, 2, 3, 4)) < (FixedArray<4, int>(2, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) < (FixedArray<4, int>(1, 3, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) < (FixedArray<4, int>(1, 2, 4, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) < (FixedArray<4, int>(1, 2, 3, 5)));

    assert((FixedArray<4, int>(2, 2, 3, 4)) > (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 3, 3, 4)) > (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 4, 4)) > (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 5)) > (FixedArray<4, int>(1, 2, 3, 4)));

    assert((FixedArray<4, int>(1, 2, 3, 4)) <= (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) <= (FixedArray<4, int>(2, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) <= (FixedArray<4, int>(1, 3, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) <= (FixedArray<4, int>(1, 2, 4, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 4)) <= (FixedArray<4, int>(1, 2, 3, 5)));

    assert((FixedArray<4, int>(1, 2, 3, 4)) >= (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(2, 2, 3, 4)) >= (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 3, 3, 4)) >= (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 4, 4)) >= (FixedArray<4, int>(1, 2, 3, 4)));
    assert((FixedArray<4, int>(1, 2, 3, 5)) >= (FixedArray<4, int>(1, 2, 3, 4)));
  }
  {
    // +=
    FixedArray<4> a(1, 2, 3, 4), b(3, 5, 8, 13);
    FixedArray<4> v(2, 3, 5, 9);
    a += v;
    assert(a == b);
  }
  {
    // -=
    FixedArray<4> a(1, 2, 3, 4), b(-1, -1, -2, -3);
    FixedArray<4> v(2, 3, 5, 7);
    a -= v;
    assert(a == b);
  }
  //
  // Math operators.
  //
  {
    // sum
    assert(computeSum(FixedArray<4>(1, 2, 3, 4)) == 10);
  }
  {
    // product
    assert(computeProduct(FixedArray<4>(1, 2, 3, 4)) == 24);
  }
  {
    // min
    assert(computeMinimum(FixedArray<4>(1, 2, 3, 4)) == 1);
    assert(computeMinimum(FixedArray<4>(2, 4, 1, 3)) == 1);
  }
  {
    // max
    assert(computeMaximum(FixedArray<4>(1, 2, 3, 4)) == 4);
    assert(computeMaximum(FixedArray<4>(2, 4, 1, 3)) == 4);
  }
  {
    // negate
    FixedArray<4> d(1, 2, 3, 4);
    d.negate();
    assert((d == FixedArray<4>(-1, -2, -3, -4)));
  }
  {
    // sort
    FixedArray<4> a(4, 3, 2, 1);
    assert(a.is_sorted(std::greater<double>()));
    a.sort();
    assert(a.is_sorted());
  }
  {
    // min_index
    {
      FixedArray<4> a(1, 2, 3, 4);
      assert(a.min_index() == 0);
    }
    {
      FixedArray<4> a(4, 3, 2 , 1);
      assert(a.min_index() == 3);
    }
  }
  {
    // max_index
    {
      FixedArray<4> a(1, 2, 3, 4);
      assert(a.max_index() == 3);
    }
    {
      FixedArray<4> a(4, 3, 2 , 1);
      assert(a.max_index() == 0);
    }
  }
  {
    // I/O
    std::stringstream file;
    file << FixedArray<4>(1, 2, 3, 4) << '\n';
    FixedArray<4> x;
    file >> x;
    assert((x == FixedArray<4>(1, 2, 3, 4)));
  }

  //
  // Standard math functions.
  //
  {
    typedef FixedArray<4> FA;
    FA a, b;
    assert(abs(FA(-1.0, 0.6, 0.7, 0.8)) ==
           FA(std::abs(-1.0), std::abs(0.6),
              std::abs(0.7), std::abs(0.8)));

    a = acos(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::acos(0.5), std::acos(0.6), std::acos(0.7), std::acos(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = asin(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::asin(0.5), std::asin(0.6), std::asin(0.7), std::asin(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = atan(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::atan(0.5), std::atan(0.6), std::atan(0.7), std::atan(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = ceil(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::ceil(0.5), std::ceil(0.6), std::ceil(0.7), std::ceil(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cos(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::cos(0.5), std::cos(0.6), std::cos(0.7), std::cos(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = cosh(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::cosh(0.5), std::cosh(0.6), std::cosh(0.7), std::cosh(0.8));
    // CONTINUE. With GCC 4.7 I get the following. The evaluation of
    // cosh() at compile time is wrong.
    // std::cerr << a << '\n' << b << '\n' << a - b << '\n';
    // 1.12763 1.18547 1.25517 1.33743
    // 1.12763 1.18547 1.25387 1.33591
    // 0 0 0.0013006 0.00152286
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = exp(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::exp(0.5), std::exp(0.6), std::exp(0.7), std::exp(0.8));
    // Likewise.
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = floor(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::floor(0.5), std::floor(0.6), std::floor(0.7),
           std::floor(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::log(0.5), std::log(0.6), std::log(0.7), std::log(0.8));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = log10(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::log10(0.5), std::log10(0.6), std::log10(0.7),
           std::log10(0.8));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sin(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::sin(0.5), std::sin(0.6), std::sin(0.7), std::sin(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sinh(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::sinh(0.5), std::sinh(0.6), std::sinh(0.7), std::sinh(0.8));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = sqrt(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::sqrt(0.5), std::sqrt(0.6), std::sqrt(0.7), std::sqrt(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tan(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::tan(0.5), std::tan(0.6), std::tan(0.7), std::tan(0.8));
    assert(areSequencesEqual(a.begin(), a.end(), b.begin()));

    a = tanh(FA(0.5, 0.6, 0.7, 0.8));
    b = FA(std::tanh(0.5), std::tanh(0.6), std::tanh(0.7), std::tanh(0.8));
    //assert(areSequencesEqual(a.begin(), a.end(), b.begin()));
  }

  return 0;
}

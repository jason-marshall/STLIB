// -*- C++ -*-

#include "stlib/ads/array/Array.h"

#include <vector>
#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  //
  // Constructor: default
  //
  {
    typedef Array<1, double> Array;
    Array a;
    assert(a.size() == 0);
    assert(a.empty());
    std::cout << "max_size = " << a.max_size() << '\n';
  }
  {
    typedef Array< 1, std::vector<double> > Array;
    Array a;
    assert(a.size() == 0);
    assert(a.empty());
  }
  //
  // Constructors: size
  //
  {
    typedef Array<1, double> Array;
    Array a(10);
    assert(a.size() == 10);
    assert(! a.empty());
    assert(a.getMemoryUsage() ==
           int(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<1, double> Array;
    Array::size_type size = 7;
    const double value = 11;
    Array a(size, value);
    assert(a.size() == size);
    assert(! a.empty());
    for (int i = 0; i < size; ++i) {
      assert(a[i] == value);
    }
  }
  //
  // Constructors: range
  //
  {
    typedef Array<1, double> Array;
    Array::range_type range(-7, 11);
    Array::size_type size = range.extent();
    Array a(range);
    assert(a.size() == size);
    assert(! a.empty());
  }
  {
    typedef Array<1, double> Array;
    Array::range_type range(-7, 11);
    Array::size_type size = range.extent();
    const double value = 11;
    Array a(range, value);
    assert(a.size() == size);
    assert(! a.empty());
    for (int i = range.lbound(); i != range.ubound(); ++i) {
      assert(a(i) == value);
    }
  }

  //
  // Construct by copying data.
  //
  {
    std::vector<double> vec(10);
    const int size = int(vec.size());
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    typedef Array<1, double> Array;
    Array a(vec.begin(), vec.end());

    // Accessors
    assert(int(a.size()) == size);
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Copy constructor.
    {
      Array b(a);
      assert(a == b);
    }

    // Assignment
    {
      Array b(a.size());
      b = a;
      assert(b == a);
    }

    // Indexing
    for (int i = 0; i < size; ++i) {
      assert(a[i] == i);
      assert(a(i) == i);
    }

    // Mathematical operations
    assert(computeSum(a) == std::accumulate(vec.begin(), vec.end(),
                                            double(0)));
    assert(computeProduct(a) == std::accumulate(vec.begin(), vec.end(),
           double(1),
           std::multiplies<double>()));
    assert(computeMinimum(a) == *std::min_element(vec.begin(), vec.end()));
    assert(computeMaximum(a) == *std::max_element(vec.begin(), vec.end()));
    a.negate();
    assert(computeSum(a) == -std::accumulate(vec.begin(), vec.end(),
           double(0)));
    a.negate();

    // I/O
    std::cout << a << '\n';
  }

  //
  // Copy constructors and assignment operators.
  //
  {
    const int size = 10;
    double data[size];
    for (int i = 0; i != size; ++i) {
      data[i] = i;
    }
    {
      Array<1, double, false> ar(size, data);
      Array<1> a(ar);
      assert(a == ar);
      Array<1> b;
      b = ar;
      assert(b == ar);
    }
    {
      Array<1, const double, false> acr(size, data);
      Array<1> a(acr);
      assert(a == acr);
      Array<1> b;
      b = acr;
      assert(b == acr);
    }
    {
      Array<3> ma(Array<3>::index_type(1, 2, 3), 7.0);
      Array<1> c(6, 7.0);
      Array<1> a(ma);
      assert(a == c);
      Array<1> b;
      b = ma;
      assert(b == c);
    }
  }

  //
  // Check assignment operators.
  //
  {
    // Construct by copying data.
    std::vector<int> vec(10);
    const int size = int(vec.size());
    for (int i = 0; i < size; ++i) {
      vec[i] = i + 1;
    }

    // Scalar operations.
    {
      Array<1, int> a(vec.begin(), vec.end());
      Array<1, int> b(a);
      int v = 2;
      a += v;
      assert(a != b);
      a -= v;
      assert(a == b);
      a *= v;
      assert(a != b);
      a /= v;
      assert(a == b);
      a %= v;
      a = v;
    }
    // Array operations.
    {
      Array<1, int> a(vec.begin(), vec.end());
      Array<1, int> b(a);
      a += b;
      assert(a != b);
      a -= b;
      assert(a == b);
      a *= b;
      assert(a != b);
      a /= b;
      assert(a == b);
      a %= b;
    }
    // Array operations with array of different type.
    {
      Array<1, int> a(vec.begin(), vec.end());
      Array<1, unsigned> b;
      b = a;
      Array<1, int> c(a);
      a += b;
      assert(a != c);
      a -= b;
      assert(a == c);
      a *= b;
      assert(a != c);
      a /= b;
      assert(a == c);
      a %= b;
    }
  }
  //
  // Resize
  //
  {
    typedef Array<1, double> Array;
    Array a(10);
    a.resize(20);
    assert(a.size() == 20);
    a.resize(Array::range_type(7, 11));
    assert(a.range() == Array::range_type(7, 11));
    assert(a.lbound() == 7);
    assert(a.ubound() == 11);
    assert(a.size() == 4);
  }
  //
  // Swap.
  //
  {
    typedef Array<1, double> Array;
    Array a(10, 1.0);
    Array ac = a;
    assert(a == ac);
    Array b(20, 2.0);
    Array bc = b;
    assert(b == bc);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  //
  // Mathematical functions.
  //
  {
    Array<1> a(1);
    const double eps = 10.0 * std::numeric_limits<double>::epsilon();

    a = -1.0;
    applyAbs(&a);
    assert(std::abs(a[0] - std::abs(-1.0)) < eps);

    a = 1.0;
    applyAcos(&a);
    assert(std::abs(a[0] - std::acos(1.0)) < eps);

    a = 1.0;
    applyAsin(&a);
    assert(std::abs(a[0] - std::asin(1.0)) < eps);

    a = 1.0;
    applyAtan(&a);
    assert(std::abs(a[0] - std::atan(1.0)) < eps);

    a = 1.5;
    applyCeil(&a);
    assert(std::abs(a[0] - std::ceil(1.5)) < eps);

    a = 1.0;
    applyCos(&a);
    assert(std::abs(a[0] - std::cos(1.0)) < eps);

    a = 1.0;
    applyCosh(&a);
    assert(std::abs(a[0] - std::cosh(1.0)) < eps);

    a = 1.0;
    applyExp(&a);
    assert(std::abs(a[0] - std::exp(1.0)) < eps);

    a = 1.5;
    applyFloor(&a);
    assert(std::abs(a[0] - std::floor(1.5)) < eps);

    a = 1.0;
    applyLog(&a);
    assert(std::abs(a[0] - std::log(1.0)) < eps);

    a = 1.0;
    applyLog10(&a);
    assert(std::abs(a[0] - std::log10(1.0)) < eps);

    a = 1.0;
    applySin(&a);
    assert(std::abs(a[0] - std::sin(1.0)) < eps);

    a = 1.0;
    applySinh(&a);
    assert(std::abs(a[0] - std::sinh(1.0)) < eps);

    a = 2.0;
    applySqrt(&a);
    assert(std::abs(a[0] - std::sqrt(2.0)) < eps);

    a = 1.0;
    applyTan(&a);
    assert(std::abs(a[0] - std::tan(1.0)) < eps);

    a = 1.0;
    applyTanh(&a);
    assert(std::abs(a[0] - std::tanh(1.0)) < eps);
  }

  //
  // I/O
  //
  {
    typedef Array<1, double> Array;
    const int size = 10;
    Array a(size);
    for (int i = 0; i != size; ++i) {
      a[i] = i;
    }

    // Ascii I/O with the standard file format.
    {
      std::stringstream file;
      file << a << '\n';
      Array x;
      file >> x;
      assert(x == a);
    }
    // Binary I/O with the standard file format.
    {
      std::stringstream file;
      a.write(file);
      Array x;
      x.read(file);
      assert(x == a);
    }
  }

  return 0;
}

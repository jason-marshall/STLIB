// -*- C++ -*-

#include "stlib/ads/array/SparseArray.h"

#include <iostream>
#include <sstream>
#include <set>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  typedef FixedArray<2, int> Index;

  {
    // Default constructor.
    SparseArray<2> x;
    assert(x.getRank() == 2);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
  }
  {
    std::set<int> primes;
    primes.insert(2);
    primes.insert(3);
    primes.insert(5);
    // Element constructor.
    const int size = 9;
    Array<1, Index> indices(size);
    indices[0] = Index(2, 2);
    indices[1] = Index(3, 2);
    indices[2] = Index(5, 2);
    indices[3] = Index(2, 3);
    indices[4] = Index(3, 3);
    indices[5] = Index(5, 3);
    indices[6] = Index(2, 5);
    indices[7] = Index(3, 5);
    indices[8] = Index(5, 5);
    double values[size];
    for (int n = 0; n != size; ++n) {
      values[n] = computeSum(indices[n]);
    }
    const double null = 0;
    SparseArray<2> x(indices.begin(), indices.end(), values, values + size,
                     null);
    assert(x.getRank() == 2);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.getNull() == null);
    assert(x.begin() + x.size() == x.end());

    assert(x(Index(-1, -1)) == null);
    assert(x(Index(0, 0)) == null);
    assert(x(Index(1, 1)) == null);

    for (int i = 0; i != size; ++i) {
      assert(x[i] == values[i]);
      assert(x(indices[i]) == values[i]);
    }

    // isNull(), isNonNull() and indexing.
    for (int i = -10; i != 10; ++i) {
      for (int j = -10; j != 10; ++j) {
        if (primes.count(i) != 0 && primes.count(j) != 0) {
          assert(x.isNonNull(Index(i, j)));
          assert(x(Index(i, j)) == i + j);
        }
        else {
          assert(x.isNull(Index(i, j)));
          assert(x(Index(i, j)) == null);
        }
      }
    }

    // fill_non_null()
    {
      typedef Array<2>::range_type range_type;
      const double a_null = -1;
      // Modified to placate gcc 3.3.
      //Array<2> a(range_type(Index(-10,-10), Index(10,10)), a_null);
      range_type range(Index(-10, -10), Index(10, 10));
      Array<2> a(range, a_null);
      x.fillNonNull(&a);

      for (int i = -10; i != 10; ++i) {
        for (int j = -10; j != 10; ++j) {
          if (primes.count(i) != 0 && primes.count(j) != 0) {
            assert(a(i, j) == i + j);
          }
          else {
            assert(a(i, j) == a_null);
          }
        }
      }
    }

    // fill()
    {
      typedef Array<2>::range_type range_type;
      const double a_null = -1;
      // Modified to placate gcc 3.3.
      //Array<2> a(range_type(Index(-10,-10), Index(10,10)), a_null - 1);
      range_type range(Index(-10, -10), Index(10, 10));
      Array<2> a(range, a_null - 1);
      x.fill(&a);

      for (int i = -10; i != 10; ++i) {
        for (int j = -10; j != 10; ++j) {
          if (primes.count(i) != 0 && primes.count(j) != 0) {
            assert(a(i, j) == i + j);
          }
          else {
            assert(a(i, j) == null);
          }
        }
      }
    }

    {
      typedef Array<2>::range_type range_type;
      // Modified to placate gcc 3.3.
      //Array<2> a(range_type(Index(10,10), Index(20,20)), null - 1);
      range_type range(Index(10, 10), Index(20, 20));
      Array<2> a(range, null - 1);
      x.fill(&a);

      for (int i = 10; i != 20; ++i) {
        for (int j = 10; j != 20; ++j) {
          if (primes.count(i) != 0 && primes.count(j) != 0) {
            assert(a(i, j) == i + j);
          }
          else {
            assert(a(i, j) == null);
          }
        }
      }
    }

    {
      typedef Array<2>::range_type range_type;
      // Modified to placate gcc 3.3.
      //Array<2> a(range_type(Index(10,0), Index(20,10)), null - 1);
      range_type range(Index(10, 0), Index(20, 10));
      Array<2> a(range, null - 1);
      x.fill(&a);

      for (int i = 10; i != 20; ++i) {
        for (int j = 0; j != 10; ++j) {
          if (primes.count(i) != 0 && primes.count(j) != 0) {
            assert(a(i, j) == i + j);
          }
          else {
            assert(a(i, j) == null);
          }
        }
      }
    }

    // Copy constructor.
    {
      SparseArray<2> y(x);
      assert(x == y);
    }

    // Assignment operator
    {
      SparseArray<2> y;
      y = x;
      assert(x == y);
    }

    // Mathematical operations
    assert(computeSum(x) == std::accumulate(values, values + size, double(0)));
    assert(computeProduct(x) == std::accumulate(values, values + size,
           double(1),
           std::multiplies<double>()));
    assert(computeMinimum(x) == *std::min_element(values, values + size));
    assert(computeMaximum(x) == *std::max_element(values, values + size));
    x.negate();
    assert(computeSum(x) == -std::accumulate(values, values + size,
           double(0)));
    x.negate();

    //
    // Swap.
    //
    {
      SparseArray<2> a(x);
      SparseArray<2> b;
      a.swap(b);
      assert(x == b);
    }

    // Ascii I/O.
    {
      std::stringstream file;
      file << x << '\n';
      SparseArray<2> a;
      file >> a;
      assert(x == a);
    }

    //
    // Mathematical functions.
    //
    {
      SparseArray<2> a(x);
      Index i(2, 2);
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
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/ads/array/SparseArray.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  //-------------------------------------------------------------------------
  // 1-D
  //-------------------------------------------------------------------------

  {
    // Default constructor.
    SparseArray<1> x;
    assert(x.getRank() == 1);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
  }
  {
    // Element constructor.
    const int size = 8;
    const int indices[size] =   {2, 3, 5, 7, 11, 13, 17, 19};
    const double values[size] = {1, 1, 2, 3, 5,  8,  13, 21};
    const double null = 0;
    SparseArray<1> x(indices, indices + size, values, values + size,
                     null);
    assert(x.getRank() == 1);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.getNull() == null);
    assert(x.begin() + x.size() == x.end());

    assert(x(-1) == null);
    assert(x(0) == null);
    assert(x(1) == null);

    for (int i = 0; i != size; ++i) {
      // Container indexing.
      assert(x[i] == values[i]);
      // Array indexing.
      assert(x(indices[i]) == values[i]);
    }

    // isNull()
    assert(x.isNull(-2) == true);
    assert(x.isNull(0) == true);
    assert(x.isNull(1) == true);
    assert(x.isNull(1000) == true);
    assert(x.isNull(2) == false);
    assert(x.isNull(3) == false);
    assert(x.isNull(5) == false);
    assert(x.isNull(19) == false);

    // isNonNull()
    assert(x.isNonNull(-2) == false);
    assert(x.isNonNull(0) == false);
    assert(x.isNonNull(1) == false);
    assert(x.isNonNull(1000) == false);
    assert(x.isNonNull(2) == true);
    assert(x.isNonNull(3) == true);
    assert(x.isNonNull(5) == true);
    assert(x.isNonNull(19) == true);

    // fillNonNull()
    {
      typedef Array<1>::range_type range_type;
      const double null = -1;
      Array<1> a(range_type(-10, 10), null);
      x.fillNonNull(&a);
      assert(a(-10) == null);
      assert(a(-9) == null);
      assert(a(-8) == null);
      assert(a(-7) == null);
      assert(a(-6) == null);
      assert(a(-5) == null);
      assert(a(-4) == null);
      assert(a(-3) == null);
      assert(a(-2) == null);
      assert(a(-1) == null);
      assert(a(0) == null);
      assert(a(1) == null);
      assert(a(2) == 1);
      assert(a(3) == 1);
      assert(a(4) == null);
      assert(a(5) == 2);
      assert(a(6) == null);
      assert(a(7) == 3);
      assert(a(8) == null);
      assert(a(9) == null);
    }
    {
      typedef Array<1>::range_type range_type;
      const double null = -1;
      Array<1> a(range_type(15, 25), null);
      x.fillNonNull(&a);
      assert(a(15) == null);
      assert(a(16) == null);
      assert(a(17) == 13);
      assert(a(18) == null);
      assert(a(19) == 21);
      assert(a(20) == null);
      assert(a(21) == null);
      assert(a(22) == null);
      assert(a(23) == null);
      assert(a(24) == null);
    }

    // fill()
    {
      typedef Array<1>::range_type range_type;
      const double null = x.getNull();
      Array<1> a(range_type(-10, 10), null - 1);
      x.fill(&a);
      assert(a(-10) == null);
      assert(a(-9) == null);
      assert(a(-8) == null);
      assert(a(-7) == null);
      assert(a(-6) == null);
      assert(a(-5) == null);
      assert(a(-4) == null);
      assert(a(-3) == null);
      assert(a(-2) == null);
      assert(a(-1) == null);
      assert(a(0) == null);
      assert(a(1) == null);
      assert(a(2) == 1);
      assert(a(3) == 1);
      assert(a(4) == null);
      assert(a(5) == 2);
      assert(a(6) == null);
      assert(a(7) == 3);
      assert(a(8) == null);
      assert(a(9) == null);
    }
    {
      typedef Array<1>::range_type range_type;
      const double null = x.getNull();
      Array<1> a(range_type(15, 25), null - 1);
      x.fill(&a);
      assert(a(15) == null);
      assert(a(16) == null);
      assert(a(17) == 13);
      assert(a(18) == null);
      assert(a(19) == 21);
      assert(a(20) == null);
      assert(a(21) == null);
      assert(a(22) == null);
      assert(a(23) == null);
      assert(a(24) == null);
    }

    // Copy constructor.
    {
      SparseArray<1> y(x);
      assert(x == y);
    }

    // Assignment operator
    {
      SparseArray<1> y;
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
      SparseArray<1> a(x);
      SparseArray<1> b;
      a.swap(b);
      assert(x == b);
    }

    // Ascii I/O.
    {
      std::stringstream file;
      file << x << '\n';
      SparseArray<1> a;
      file >> a;
      assert(x == a);
    }

    //
    // Mathematical functions.
    //
    {
      SparseArray<1> a(x);
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

  //
  // Count the non-null elements.
  //
  {
    SparseArray<1> x, y;
    assert(countNonNullElementsInUnion(x, y) == 0);
  }
  {
    const int size = 8;
    const int indices[size] =   { 2, 3, 5, 7, 11, 13, 17, 19 };
    const double values[size] = { 1, 1, 2, 3, 5,  8,  13, 21 };
    const double null = 0;
    SparseArray<1> x(indices, indices + size, values, values + size,
                     null);
    SparseArray<1> y;
    assert(countNonNullElementsInUnion(x, y) == size);
    assert(countNonNullElementsInUnion(y, x) == size);
  }
  {
    const int x_size = 4;
    const int x_indices[x_size] =   { 2, 3, 5, 7};
    const double x_values[x_size] = { 1, 1, 1, 1};
    const double null = 0;
    SparseArray<1> x(x_indices, x_indices + x_size,
                     x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 2, 3, 5, 7};
    const double y_values[y_size] = { 1, 1, 1, 1};
    SparseArray<1> y(y_indices, y_indices + y_size,
                     y_values, y_values + y_size, null);

    assert(countNonNullElementsInUnion(x, y) == 4);
  }
  {
    const int x_size = 4;
    const int x_indices[x_size] =   { 2, 3, 5, 7};
    const double x_values[x_size] = { 1, 1, 1, 1};
    const double null = 0;
    SparseArray<1> x(x_indices, x_indices + x_size,
                     x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 1, 2, 3, 4};
    const double y_values[y_size] = { 1, 1, 1, 1};
    SparseArray<1> y(y_indices, y_indices + y_size,
                     y_values, y_values + y_size, null);

    assert(countNonNullElementsInUnion(x, y) == 6);
    assert(countNonNullElementsInUnion(y, x) == 6);
  }

  {
    const int x_size = 4;
    const int x_indices[x_size] =   { 2, 3, 5, 7};
    const double x_values[x_size] = { 1, 1, 1, 1};
    const double null = 0;
    SparseArray<1> x(x_indices, x_indices + x_size,
                     x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 11, 12, 13, 14};
    const double y_values[y_size] = { 1, 1, 1, 1};
    SparseArray<1> y(y_indices, y_indices + y_size,
                     y_values, y_values + y_size, null);

    assert(countNonNullElementsInUnion(x, y) == 8);
    assert(countNonNullElementsInUnion(y, x) == 8);
  }

  //
  // Sum, difference, and product of arrays.
  //
  {
    ads::Array<1> a(3), b(3), c(3);
    a[0] = 2;
    a[1] = 3;
    a[2] = 5;
    b[0] = 2;
    b[1] = 4;
    b[2] = 6;

    // Sum
    {
      c = a;
      c += b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeSum(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
    // Difference.
    {
      c = a;
      c -= b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeDifference(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
    // Product
    {
      c = a;
      c *= b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeProduct(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
  }
  {
    ads::Array<1> a(3), b(3), c(3);
    a[0] = 1;
    a[1] = 0;
    a[2] = 0;
    b[0] = 0;
    b[1] = 0;
    b[2] = 1;

    // Sum
    {
      c = a;
      c += b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeSum(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
    // Difference.
    {
      c = a;
      c -= b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeDifference(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
    // Product
    {
      c = a;
      c *= b;
      SparseArray<1> x(a, 0), y(b, 0), z;
      ads::computeProduct(x, y, &z);
      assert(z == SparseArray<1>(c, 0));
    }
  }
  //
  // Operations with arrays and sparse arrays.
  //
  {
    ads::Array<1, int> x(3), initial(3), argument(3), result(3);
    initial[0] = 2;
    initial[1] = 3;
    initial[2] = 5;
    argument[0] = 1;
    argument[1] = 0;
    argument[2] = 3;

    // Sum
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 3;
      result[1] = 3;
      result[2] = 8;
      x = initial;
      x += sparse;
      assert(x == result);
    }
    // Difference.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 1;
      result[1] = 3;
      result[2] = 2;
      x = initial;
      x -= sparse;
      assert(x == result);
    }
    // Product.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 2;
      result[1] = 3;
      result[2] = 15;
      x = initial;
      x *= sparse;
      assert(x == result);
    }
    // Quotient.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 2;
      result[1] = 3;
      result[2] = 1;
      x = initial;
      x /= sparse;
      assert(x == result);
    }
    // Remainder.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 0;
      result[1] = 3;
      result[2] = 2;
      x = initial;
      x %= sparse;
      assert(x == result);
    }
    // Scale and add.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 4;
      result[1] = 3;
      result[2] = 11;
      x = initial;
      ads::scaleAdd(&x, 2, sparse);
      assert(x == result);
    }
  }

  //
  // Operations with vectors and sparse arrays.
  //
  {
    std::vector<int> x(3), initial(3), argument(3), result(3);
    initial[0] = 2;
    initial[1] = 3;
    initial[2] = 5;
    argument[0] = 1;
    argument[1] = 0;
    argument[2] = 3;

    // Sum
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 3;
      result[1] = 3;
      result[2] = 8;
      x = initial;
      x += sparse;
      assert(x == result);
    }
    // Difference.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 1;
      result[1] = 3;
      result[2] = 2;
      x = initial;
      x -= sparse;
      assert(x == result);
    }
    // Product.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 2;
      result[1] = 3;
      result[2] = 15;
      x = initial;
      x *= sparse;
      assert(x == result);
    }
    // Quotient.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 2;
      result[1] = 3;
      result[2] = 1;
      x = initial;
      x /= sparse;
      assert(x == result);
    }
    // Remainder.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 0;
      result[1] = 3;
      result[2] = 2;
      x = initial;
      x %= sparse;
      assert(x == result);
    }
    // Scale and add.
    {
      ads::SparseArray<1, int> sparse(argument, 0);
      result[0] = 4;
      result[1] = 3;
      result[2] = 11;
      x = initial;
      scaleAdd(&x, 2, sparse);
      assert(x == result);
    }
  }
  return 0;
}

// -*- C++ -*-

// CONTINUE
//#include "stlib/ads/array/SparseArraySigned.h"

#include <iostream>
#include <sstream>

#include <cassert>

int
main()
{
  // CONTINUE
#if 0

  using namespace ads;

  //-------------------------------------------------------------------------
  // 1-D
  //-------------------------------------------------------------------------

  {
    // Default constructor.
    SparseArraySigned<1> x;
    assert(x.rank() == 1);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.memory() > 0);
  }
  {
    // Element constructor.
    const int size = 8;
    const int indices[size] =   { 2, 3, 5, 7, 11, 13, 17, 19 };
    const double values[size] = { 1, 1, 2, 3, 5,  8,  13, 21 };
    const double null = 0;
    SparseArraySigned<1> x(indices, indices + size, values, values + size,
                           null);
    assert(x.rank() == 1);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.memory() > 0);
    assert(x.null() == null);
    assert(x.begin() + x.size() == x.end());

    assert(x[ -1 ] == null);
    assert(x[ 0 ] == null);
    assert(x[ 1 ] == null);

    for (int i = 0; i != size; ++i) {
      assert(x[ indices[i] ] == values[i]);
    }

    // is_null()
    assert(x.is_null(-2) == true);
    assert(x.is_null(0) == true);
    assert(x.is_null(1) == true);
    assert(x.is_null(1000) == true);
    assert(x.is_null(2) == false);
    assert(x.is_null(3) == false);
    assert(x.is_null(5) == false);
    assert(x.is_null(19) == false);

    // is_non_null()
    assert(x.is_non_null(-2) == false);
    assert(x.is_non_null(0) == false);
    assert(x.is_non_null(1) == false);
    assert(x.is_non_null(1000) == false);
    assert(x.is_non_null(2) == true);
    assert(x.is_non_null(3) == true);
    assert(x.is_non_null(5) == true);
    assert(x.is_non_null(19) == true);

    // fill_non_null()
    {
      typedef Array<1>::range_type range_type;
      const double null = -1;
      Array<1> a(range_type(-10, 10), null);
      x.fill_non_null(a);
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
      x.fill_non_null(a);
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
      const double null = x.null();
      Array<1> a(range_type(-10, 10), null - 1);
      x.fill(a);
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
      const double null = x.null();
      Array<1> a(range_type(15, 25), null - 1);
      x.fill(a);
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
      SparseArraySigned<1> y(x);
      assert(x == y);
    }

    // Assignment operator
    {
      SparseArraySigned<1> y;
      y = x;
      assert(x == y);
    }

    // Mathematical operations
    assert(sum(x) == std::accumulate(values, values + size, double(0)));
    assert(product(x) == std::accumulate(values, values + size, double(1),
                                         std::multiplies<double>()));
    assert(min(x) == *std::min_element(values, values + size));
    assert(max(x) == *std::max_element(values, values + size));
    x.negate();
    assert(sum(x) == -std::accumulate(values, values + size, double(0)));
    x.negate();

    //
    // Swap.
    //
    {
      SparseArraySigned<1> a(x);
      SparseArraySigned<1> b;
      a.swap(b);
      assert(x == b);
    }

    // Ascii I/O.
    {
      std::stringstream file;
      file << x << '\n';
      SparseArraySigned<1> a;
      file >> a;
      assert(x == a);
    }

    //
    // Mathematical functions.
    //
    {
      SparseArraySigned<1> a(x);
      const double eps = 10.0 * std::numeric_limits<double>::epsilon();

      a = -1.0;
      a.abs();
      assert(std::abs(a[2] - std::abs(-1.0)) < eps);

      a = 1.0;
      a.acos();
      assert(std::abs(a[2] - std::acos(1.0)) < eps);

      a = 1.0;
      a.asin();
      assert(std::abs(a[2] - std::asin(1.0)) < eps);

      a = 1.0;
      a.atan();
      assert(std::abs(a[2] - std::atan(1.0)) < eps);

      a = 1.5;
      a.ceil();
      assert(std::abs(a[2] - std::ceil(1.5)) < eps);

      a = 1.0;
      a.cos();
      assert(std::abs(a[2] - std::cos(1.0)) < eps);

      a = 1.0;
      a.cosh();
      assert(std::abs(a[2] - std::cosh(1.0)) < eps);

      a = 1.0;
      a.exp();
      assert(std::abs(a[2] - std::exp(1.0)) < eps);

      a = 1.5;
      a.floor();
      assert(std::abs(a[2] - std::floor(1.5)) < eps);

      a = 1.0;
      a.log();
      assert(std::abs(a[2] - std::log(1.0)) < eps);

      a = 1.0;
      a.log10();
      assert(std::abs(a[2] - std::log10(1.0)) < eps);

      a = 1.0;
      a.sin();
      assert(std::abs(a[2] - std::sin(1.0)) < eps);

      a = 1.0;
      a.sinh();
      assert(std::abs(a[2] - std::sinh(1.0)) < eps);

      a = 2.0;
      a.sqrt();
      assert(std::abs(a[2] - std::sqrt(2.0)) < eps);

      a = 1.0;
      a.tan();
      assert(std::abs(a[2] - std::tan(1.0)) < eps);

      a = 1.0;
      a.tanh();
      assert(std::abs(a[2] - std::tanh(1.0)) < eps);
    }
  }

  {
    // Element constructor.
    const int size = 7;
    const int indices[size] =   { -1, 0, 1,  5, 6,   10, 11 };
    const double values[size] = { -1, 0, 1,  1, -1,  -1,  1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(indices, indices + size, values, values + size,
                           null);

    assert(x(-2) == -null);
    assert(x(-1) == - 1);
    assert(x(0) == 0);
    assert(x(1) == 1);
    assert(x(2) == null);
    assert(x(3) == null);
    assert(x(4) == null);
    assert(x(5) == 1);
    assert(x(6) == -1);
    assert(x(7) == -null);
    assert(x(8) == -null);
    assert(x(9) == -null);
    assert(x(10) == -1);
    assert(x(11) == 1);
    assert(x(12) == null);

    {
      Array<1> a(IndexRange<1>(-2, 13));
      x.fill(a);
      assert(a(-2) == -null);
      assert(a(-1) == - 1);
      assert(a(0) == 0);
      assert(a(1) == 1);
      assert(a(2) == null);
      assert(a(3) == null);
      assert(a(4) == null);
      assert(a(5) == 1);
      assert(a(6) == -1);
      assert(a(7) == -null);
      assert(a(8) == -null);
      assert(a(9) == -null);
      assert(a(10) == -1);
      assert(a(11) == 1);
      assert(a(12) == null);
    }

    {
      Array<1> a(IndexRange<1>(-10, -8));
      x.fill(a);
      assert(a(-10) == -null);
      assert(a(-9) == -null);
    }

    {
      Array<1> a(IndexRange<1>(20, 22));
      x.fill(a);
      assert(a(20) == null);
      assert(a(21) == null);
    }
  }

  //
  // Merge.
  //
  {
    SparseArraySigned<1> x, y, z;
    merge(x, y, z);
    assert(z.empty());
  }
  {
    const int size = 8;
    const int indices[size] =   { 2, 3, 5, 7, 11, 13, 17, 19 };
    const double values[size] = { 1, 1, 2, 3, 5,  8,  13, 21 };
    const double null = 0;
    SparseArraySigned<1> x(indices, indices + size, values, values + size,
                           null);
    SparseArraySigned<1> y(null), z;
    merge(x, y, z);
    assert(x == z);
    merge(y, x, z);
    assert(x == z);
    assert(z(2) == 1);
    assert(z(3) == 1);
    assert(z(5) == 2);
    assert(z(7) == 3);
    assert(z(11) == 5);
    assert(z(13) == 8);
    assert(z(17) == 13);
    assert(z(19) == 21);
  }
  {
    const int x_size = 4;
    const int x_indices[x_size] =   { 2, 3, 5, 7};
    const double x_values[x_size] = { 1, 1, 1, 1};
    const double null = 0;
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 2, 3, 5, 7};
    const double y_values[y_size] = { 1, 1, 1, 1};
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;
    merge(x, y, z);
    assert(x == z);
    assert(z(2) == 1);
    assert(z(3) == 1);
    assert(z(5) == 1);
    assert(z(7) == 1);
  }
  {
    const int x_size = 4;
    const int x_indices[x_size] =   { 1, 2, 3, 4};
    const double x_values[x_size] = { 2, 4, 6, 8};
    const double null = 0;
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 3, 4, 5, 6};
    const double y_values[y_size] = { 6, 8, 10, 12};
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;
    merge(x, y, z);
    assert(z.size() == 6);
    assert(z(1) == 2);
    assert(z(2) == 4);
    assert(z(3) == 6);
    assert(z(4) == 8);
    assert(z(5) == 10);
    assert(z(6) == 12);
  }

  //--------------------------------------------------------------------------
  // Union.
  //--------------------------------------------------------------------------

  {
    SparseArraySigned<1> x, y, z;

    x.set_sign(1);
    y.set_sign(1);
    compute_union(x, y, z);
    assert(z.empty() && z.sign() == 1);

    x.set_sign(1);
    y.set_sign(-1);
    compute_union(x, y, z);
    assert(z.empty() && z.sign() == -1);

    x.set_sign(-1);
    y.set_sign(1);
    compute_union(x, y, z);
    assert(z.empty() && z.sign() == -1);

    x.set_sign(-1);
    y.set_sign(-1);
    compute_union(x, y, z);
    assert(z.empty() && z.sign() == -1);
  }

  {
    // Disjoint.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 20, 21, 29, 30 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 2 * size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(20) == 1);
    assert(z(21) == -1);
    assert(z(29) == -1);
    assert(z(30) == 1);

    compute_union(y, x, z);
    assert(z.size() == 2 * size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(20) == 1);
    assert(z(21) == -1);
    assert(z(29) == -1);
    assert(z(30) == 1);
  }

  {
    // They are close.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 10, 11, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 2 * size - 1);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(11) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);

    compute_union(y, x, z);
    assert(z.size() == 2 * size - 1);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(11) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);
  }
  {
    // They touch, one cell overlap.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 10, 11, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 7);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(11) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);


    compute_union(y, x, z);
    assert(z.size() == 7);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
    assert(z(11) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);
  }
  {
    // They touch, two cell overlap.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 9, 10, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);


    compute_union(y, x, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(19) == -1);
    assert(z(20) == 1);
  }
  {
    // They intersect.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 5, 6, 15, 16 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(15) == -1);
    assert(z(16) == 1);

    compute_union(y, x, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(15) == -1);
    assert(z(16) == 1);
  }

  {
    // y is a subset of x.  Both finite.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 4, 5, 6, 7 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);

    compute_union(y, x, z);
    assert(z.size() == size);
    assert(z(0) == 1);
    assert(z(1) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
  }

  {
    // y is a subset of x.  x is infinite.
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(null);
    x.set_sign(-1);
    const int size = 4;
    const int y_indices[size] =   { 4, 5, 6, 7 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 0);
    assert(z.sign() == -1);

    compute_union(y, x, z);
    assert(z.size() == 0);
    assert(z.sign() == -1);
  }

  {
    // y is a subset of x.  x is semi-infinite.
    const double null = std::numeric_limits<double>::max();
    const int x_size = 2;
    const int x_indices[x_size] =   { 0, 1 };
    const double x_values[x_size] = { 1, -1 };
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 4, 5, 6, 7 };
    const double y_values[y_size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 2);
    assert(z(0) == 1);
    assert(z(1) == -1);

    compute_union(y, x, z);
    assert(z.size() == 2);
    assert(z(0) == 1);
    assert(z(1) == -1);
  }

  {
    // y is a subset of !x.  x is semi-infinite.
    const double null = std::numeric_limits<double>::max();
    const int x_size = 2;
    const int x_indices[x_size] =   { 0, 1 };
    const double x_values[x_size] = { -1, 1 };
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 4, 5, 6, 7 };
    const double y_values[y_size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;

    compute_union(x, y, z);
    assert(z.size() == 6);
    assert(z(0) == -1);
    assert(z(1) == 1);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);

    compute_union(y, x, z);
    assert(z.size() == 6);
    assert(z(0) == -1);
    assert(z(1) == 1);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);
  }

  //--------------------------------------------------------------------------
  // Intersection.
  //--------------------------------------------------------------------------

  {
    SparseArraySigned<1> x, y, z;

    x.set_sign(1);
    y.set_sign(1);
    compute_intersection(x, y, z);
    assert(z.empty() && z.sign() == 1);

    x.set_sign(1);
    y.set_sign(-1);
    compute_intersection(x, y, z);
    assert(z.empty() && z.sign() == 1);

    x.set_sign(-1);
    y.set_sign(1);
    compute_intersection(x, y, z);
    assert(z.empty() && z.sign() == 1);

    x.set_sign(-1);
    y.set_sign(-1);
    compute_intersection(x, y, z);
    assert(z.empty() && z.sign() == -1);
  }

  {
    // Disjoint.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 20, 21, 29, 30 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);

    compute_intersection(y, x, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);
  }
  {
    // They are close.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 10, 11, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);

    compute_intersection(y, x, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);
  }
  {
    // They touch, one cell overlap.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 10, 11, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);

    compute_intersection(y, x, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);
  }
  {
    // They touch, two cell overlap.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 9, 10, 19, 20 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);

    compute_intersection(y, x, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);
  }
  {
    // They intersect.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 5, 6, 15, 16 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == size);
    assert(z(5) == 1);
    assert(z(6) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);

    compute_intersection(y, x, z);
    assert(z.size() == size);
    assert(z(5) == 1);
    assert(z(6) == -1);
    assert(z(9) == -1);
    assert(z(10) == 1);
  }
  {
    // y is a subset of x.  Both finite.
    const int size = 4;
    const int x_indices[size] =   { 0, 1, 9, 10 };
    const double x_values[size] = { 1, -1, -1, 1 };
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(x_indices, x_indices + size,
                           x_values, x_values + size, null);
    const int y_indices[size] =   { 4, 5, 6, 7 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);

    compute_intersection(y, x, z);
    assert(z.size() == size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);
  }
  {
    // y is a subset of x.  x is infinite.
    const double null = std::numeric_limits<double>::max();
    SparseArraySigned<1> x(null);
    x.set_sign(-1);
    const int size = 4;
    const int y_indices[size] =   { 4, 5, 6, 7 };
    const double y_values[size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + size,
                           y_values, y_values + size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);

    compute_intersection(y, x, z);
    assert(z.size() == size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);
  }
  {
    // y is a subset of x.  x is semi-infinite.
    const double null = std::numeric_limits<double>::max();
    const int x_size = 2;
    const int x_indices[x_size] =   { 0, 1 };
    const double x_values[x_size] = { 1, -1 };
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 4, 5, 6, 7 };
    const double y_values[y_size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == y_size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);

    compute_intersection(y, x, z);
    assert(z.size() == y_size);
    assert(z(4) == 1);
    assert(z(5) == -1);
    assert(z(6) == -1);
    assert(z(7) == 1);
  }
  {
    // y is a subset of !x.  x is semi-infinite.
    const double null = std::numeric_limits<double>::max();
    const int x_size = 2;
    const int x_indices[x_size] =   { 0, 1 };
    const double x_values[x_size] = { -1, 1 };
    SparseArraySigned<1> x(x_indices, x_indices + x_size,
                           x_values, x_values + x_size, null);
    const int y_size = 4;
    const int y_indices[y_size] =   { 4, 5, 6, 7 };
    const double y_values[y_size] = { 1, -1, -1, 1 };
    SparseArraySigned<1> y(y_indices, y_indices + y_size,
                           y_values, y_values + y_size, null);

    SparseArraySigned<1> z;

    compute_intersection(x, y, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);

    compute_intersection(y, x, z);
    assert(z.size() == 0);
    assert(z.empty() && z.sign() == 1);
  }

#endif

  return 0;
}

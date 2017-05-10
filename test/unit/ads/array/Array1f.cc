// -*- C++ -*-

//
// Tests for Array<1,T,false>.
//

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
    Array<1, double, false> a;
    assert(a.size() == 0);
    assert(a.empty());
  }
  {
    Array<1, const double, false> a;
    assert(a.size() == 0);
    assert(a.empty());
  }

  //
  // Construct by referencing data.
  //
  {
    const int size = 10;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    Array<1, double, false> a(int(vec.size()), &*vec.begin());

    // Accessors
    assert(int(a.size()) == size);
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Copy constructor.
    {
      Array<1, double, false> b(a);
      assert(a == b);
    }

    // Assignment
    {
      Array<1, double, false> b;
      b = a;
      assert(b == a);
    }

    // Indexing
    for (int i = 0; i < size; ++i) {
      assert(a[i] == i);
      assert(a(i) == i);
    }

    // Mathematical operations
    const double array_sum = std::accumulate(vec.begin(), vec.end(),
                             double(0));
    assert(computeSum(a) == array_sum);
    assert(computeProduct(a) == std::accumulate(vec.begin(), vec.end(),
           double(1),
           std::multiplies<double>()));
    assert(computeMinimum(a) == *std::min_element(vec.begin(), vec.end()));
    assert(computeMaximum(a) == *std::max_element(vec.begin(), vec.end()));
    a.negate();
    assert(computeSum(a) == - array_sum);
    a.negate();
  }
  {
    const int size = 10;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    Array<1, const double, false> a(int(vec.size()), &*vec.begin());

    // Accessors
    assert(a.size() == size);
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Copy constructor.
    {
      Array<1, const double, false> b(a);
      assert(a == b);
    }

    // Assignment
    {
      Array<1, const double, false> b;
      b = a;
      assert(b == a);
    }

    // Indexing
    for (int i = 0; i < size; ++i) {
      assert(a[i] == i);
      assert(a(i) == i);
    }

    // Mathematical operations
    const double array_sum = std::accumulate(vec.begin(), vec.end(),
                             double(0));
    assert(computeSum(a) == array_sum);
    assert(computeProduct(a) == std::accumulate(vec.begin(), vec.end(),
           double(1),
           std::multiplies<double>()));
    assert(computeMinimum(a) == *std::min_element(vec.begin(), vec.end()));
    assert(computeMaximum(a) == *std::max_element(vec.begin(), vec.end()));
  }

  //
  // Copy constructors and assignment operators.
  //
  {
    {
      Array<1> a(10, 23.0);
      Array<1, double, false> b(a);
      assert(a == b);
      Array<1, double, false> c;
      c = a;
      assert(a == c);
    }
    {
      Array<3> a(Array<3>::index_type(1, 2, 3), 7.0);
      Array<1> values(6, 7.0);
      Array<1, double, false> b(a);
      assert(b == values);
      Array<1, double, false> c;
      c = a;
      assert(c == values);
    }
  }
  {
    {
      Array<1> a(10, 23.0);
      Array<1, const double, false> b(a);
      assert(a == b);
      Array<1, const double, false> c;
      c = a;
      assert(a == c);
    }
    {
      Array<3> a(Array<3>::index_type(1, 2, 3), 7.0);
      Array<1> values(6, 7.0);
      Array<1, const double, false> b(a);
      assert(b == values);
      Array<1, const double, false> c;
      c = a;
      assert(c == values);
    }
  }

  //
  // Check assignment operators.
  //
  {
    const int size = 10;

    // Scalar operations.
    {
      std::vector<int> vec(size);
      for (int i = 0; i < size; ++i) {
        vec[i] = i + 1;
      }

      Array<1, int, false> a(int(vec.size()), &*vec.begin());
      Array<1, int> b;
      b = a;
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
      std::vector<int> vec(size);
      for (int i = 0; i < size; ++i) {
        vec[i] = i + 1;
      }

      Array<1, int, false> a(int(vec.size()), &*vec.begin());
      Array<1, int> b;
      b = a;
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
      std::vector<int> vec(size);
      for (int i = 0; i < size; ++i) {
        vec[i] = i + 1;
      }

      Array<1, int, false> a(int(vec.size()), &*vec.begin());
      Array<1, unsigned> b;
      b = a;
      Array<1, int> c;
      c = a;
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
  // Swap.
  //
  {
    typedef Array<1, double, false> ArrayRef;

    const int a_size = 10;
    double a_array[a_size];
    ArrayRef a(a_size, a_array);
    for (int i = 0; i != a_size; ++i) {
      a[i] = i;
    }

    const int b_size = 20;
    double b_array[b_size];
    ArrayRef b(b_size, b_array);
    for (int i = 0; i != b_size; ++i) {
      b[i] = i;
    }

    ArrayRef ac = a;
    assert(a == ac);

    ArrayRef bc = b;
    assert(b == bc);

    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    typedef Array<1, const double, false> ArrayRef;

    const int a_size = 10;
    double a_array[a_size];
    ArrayRef a(a_size, a_array);
    for (int i = 0; i != a_size; ++i) {
      a_array[i] = i;
    }

    const int b_size = 20;
    double b_array[b_size];
    ArrayRef b(b_size, b_array);
    for (int i = 0; i != b_size; ++i) {
      b_array[i] = i;
    }

    ArrayRef ac = a;
    assert(a == ac);

    ArrayRef bc = b;
    assert(b == bc);

    a.swap(b);
    assert(a == bc && b == ac);
  }
  //
  // I/O
  //
  {
    typedef Array<1, double, false> ArrayRef;
    const int size = 10;
    double a_array[size];
    ArrayRef a(size, a_array);
    for (int i = 0; i != size; ++i) {
      a[i] = i;
    }

    // Ascii I/O with the standard file format.
    {
      std::stringstream file;
      file << a << '\n';
      double x_array[size];
      ArrayRef x(size, x_array);
      file >> x;
      assert(x == a);
    }
    // Binary I/O with the standard file format.
    {
      std::stringstream file;
      a.write(file);
      double x_array[size];
      ArrayRef x(size, x_array);
      x.read(file);
      assert(x == a);
    }
  }

  return 0;
}

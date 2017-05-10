// -*- C++ -*-

//
// Tests for Array<N,T,false>.
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
  //
  // Constructor: default
  //
  {
    typedef ads::Array<1, double, false> Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    Array a;

    assert(a.size() == 0);

    index_type extents0;
    extents0 = 0;
    assert(a.extents() == extents0);
    assert(a.extent(0) == 0);

    index_type index0;
    index0 = 0;
    assert(a.lbounds() == index0);
    assert(a.ubounds() == index0);
    assert(a.ranges() == range_type(index0, index0));

    assert(a.empty());
  }
  {
    typedef ads::Array<2, double, false> Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    Array a;
    assert(a.size() == 0);

    assert(a.extents() == index_type(0, 0));
    assert(a.extent(0) == 0 && a.extent(1) == 0);

    assert(a.lbounds() == index_type(0, 0));
    assert(a.ubounds() == index_type(0, 0));
    assert(a.ranges() == range_type(index_type(0, 0),
                                    index_type(0, 0)));

    assert(a.empty());
  }
  {
    typedef ads::Array< 2, std::vector<int>, false > Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    Array a;
    assert(a.size() == 0);

    assert(a.extents() == index_type(0, 0));
    assert(a.extent(0) == 0 && a.extent(1) == 0);

    assert(a.lbounds() == index_type(0, 0));
    assert(a.ubounds() == index_type(0, 0));
    assert(a.ranges() == range_type(index_type(0, 0),
                                    index_type(0, 0)));

    assert(a.empty());
  }
  {
    typedef ads::Array<3, double, false> Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    Array a;
    assert(a.size() == 0);

    assert(a.extents() == index_type(0, 0, 0));
    assert(a.extent(0) == 0 && a.extent(1) == 0 &&  a.extent(2) == 0);

    assert(a.lbounds() == index_type(0, 0, 0));
    assert(a.ubounds() == index_type(0, 0, 0));
    assert(a.ranges() == range_type(index_type(0, 0, 0),
                                    index_type(0, 0, 0)));

    assert(a.empty());
  }

  //
  // Copy constructor.
  //
  {
    typedef ads::Array<3, double, false> ArrayRef;
    typedef ads::Array<3> Array;
    typedef Array::index_type index_type;
    Array a(index_type(10, 20, 30), 42);
    ArrayRef b(a);
    assert(b == a);
  }

  //
  // Extents and pointer constructor.
  //
  {
    const int size = 24;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    typedef ads::Array<3, double, false> Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;
    index_type multi_size(2, 3, 4);
    index_type lb, ub;
    lb = 0;
    ub = multi_size;
    Array a(multi_size, &*vec.begin());

    // Accessors
    assert(a.size() == computeProduct(multi_size));
    assert(a.extents() == multi_size);
    assert(a.extent(0) == multi_size[0] &&
           a.extent(1) == multi_size[1] &&
           a.extent(2) == multi_size[2]);
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Indexing
    {
      index_type mi;
      int i = 0;
      for (mi[2] = a.lbounds()[2]; mi[2] != a.ubounds()[2]; ++mi[2]) {
        for (mi[1] = a.lbounds()[1]; mi[1] != a.ubounds()[1]; ++mi[1]) {
          for (mi[0] = a.lbounds()[0]; mi[0] != a.ubounds()[0]; ++mi[0]) {
            assert(a[i] == i);
            assert(a(mi) == vec[i]);
            assert(a(mi[0], mi[1], mi[2]) == a(mi));
            ++i;
          }
        }
      }
    }

    // Mathematical operations
    const double array_sum =
      std::accumulate(vec.begin(), vec.end(), double(0));
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

  //
  // Index range and pointer constructor.
  //
  {
    const int size = 24;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    typedef ads::Array<3, double, false> Array;
    typedef Array::index_type index_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;
    index_type lb(2, 3 , 5), ub(4, 6, 9);
    index_type multi_size;
    multi_size = ub - lb;
    Array a(range_type(lb, ub), &*vec.begin());

    // Accessors
    assert(a.size() == computeProduct(multi_size));
    assert(a.extents() == multi_size);
    assert(a.extent(0) == multi_size[0] &&
           a.extent(1) == multi_size[1] &&
           a.extent(2) == multi_size[2]);
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Indexing
    {
      index_type mi;
      int i = 0;
      for (mi[2] = a.lbounds()[2]; mi[2] != a.ubounds()[2]; ++mi[2]) {
        for (mi[1] = a.lbounds()[1]; mi[1] != a.ubounds()[1]; ++mi[1]) {
          for (mi[0] = a.lbounds()[0]; mi[0] != a.ubounds()[0]; ++mi[0]) {
            assert(a(mi) == vec[i]);
            assert(a(mi[0], mi[1], mi[2]) == a(mi));
            ++i;
          }
        }
      }
    }

    // Mathematical operations
    const double array_sum =
      std::accumulate(vec.begin(), vec.end(), double(0));
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

  //
  // Assignment operators
  //
  {
    const int size = 24;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    typedef ads::Array<3>::index_type index_type;
    ads::Array<3, double, false> a(index_type(2, 3, 4), &*vec.begin());

    // Assignment to Array.
    {
      ads::Array<3> b;
      b = a;
      assert(a == b);
    }
    // Assignment to value.
    {
      double mem[size];
      ads::Array<3, double, false> b(index_type(2, 3, 4), mem);
      b = 7;
      for (ads::Array<3, double, false>::const_iterator i = b.begin();
           i != b.end(); ++i) {
        assert(*i == 7);
      }
    }
    // Assignment to Array.
    {
      ads::Array<3> b(index_type(2, 3, 4),
                      vec.begin(), vec.end());
      ads::Array<3, double, false> c;
      c = b;
      assert(c == b);
    }
  }

  //
  // Swap.
  //
  {
    typedef ads::Array<2, double, false> Array;
    typedef Array::index_type index_type;

    const int a_size = 10;
    const index_type a_extents(2, 5);
    double a_array[a_size];
    Array a(a_extents, a_array);
    for (int i = 0; i != a_size; ++i) {
      a[i] = i;
    }

    const int b_size = 20;
    const index_type b_extents(4, 5);
    double b_array[b_size];
    Array b(b_extents, b_array);
    for (int i = 0; i != b_size; ++i) {
      b[i] = i;
    }

    Array ac = a;
    assert(a == ac);

    Array bc = b;
    assert(b == bc);

    a.swap(b);
    assert(a == bc && b == ac);
  }
  {
    typedef ads::Array<2, const double, false> Array;
    typedef Array::index_type index_type;

    const int a_size = 10;
    const index_type a_extents(2, 5);
    double a_array[a_size];
    Array a(a_extents, a_array);
    for (int i = 0; i != a_size; ++i) {
      a_array[i] = i;
    }

    const int b_size = 20;
    const index_type b_extents(4, 5);
    double b_array[b_size];
    Array b(b_extents, b_array);
    for (int i = 0; i != b_size; ++i) {
      b_array[i] = i;
    }

    Array ac = a;
    assert(a == ac);

    Array bc = b;
    assert(b == bc);

    a.swap(b);
    assert(a == bc && b == ac);
  }

  //
  // I/O
  //
  {
    typedef ads::Array<3, double, false> ArrayRef;
    typedef ArrayRef::index_type index_type;

    const index_type extents(2, 3, 4);
    const int size = computeProduct(extents);
    double* a_array = new double[ size ];
    ArrayRef a(extents, a_array);
    for (int i = 0; i != size; ++i) {
      a[i] = i;
    }

    // Ascii I/O with the standard file format.
    {
      std::stringstream file;
      file << a << '\n';
      double* x_array = new double[ size ];
      ArrayRef x(extents, x_array);
      file >> x;
      assert(x == a);
      delete[] x_array;
    }
    // Binary I/O with the standard file format.
    {
      std::stringstream file;
      a.write(file);
      double* x_array = new double[ size ];
      ArrayRef x(extents, x_array);
      x.read(file);
      assert(x == a);
      delete[] x_array;
    }
    delete[] a_array;
  }

  return 0;
}

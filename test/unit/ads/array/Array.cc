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
    typedef Array<1> Array;
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
    typedef Array<2> Array;
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
    typedef Array< 2, std::vector<int> > Array;
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
    typedef Array<3> Array;
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
  // Constructor: extents
  //
  {
    typedef Array<1> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type extents;
    extents = 10;
    Array a(extents);

    assert(a.size() == 10);

    assert(a.extents() == extents);
    assert(a.extent(0) == 10);

    index_type lb, ub;
    lb = 0;
    ub = extents;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<1> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type extents(10);
    size_type extent = 10;
    Array a(extent);

    assert(a.size() == 10);

    assert(a.extents() == extents);
    assert(a.extent(0) == extent);

    index_type lb, ub;
    lb = 0;
    ub = extents;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<2> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type size(10, 20);
    Array a(size);

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]);

    index_type lb(0, 0), ub;
    ub = size;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<2> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    size_type extent0 = 10;
    size_type extent1 = 20;
    index_type extents(extent0, extent1);
    Array a(extents);

    assert(a.size() == computeProduct(extents));

    assert(a.extents() == extents);
    assert(a.extent(0) == extent0 && a.extent(1) == extent1);

    index_type lb(0, 0), ub;
    ub = extents;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<3> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type size(10, 20, 30);
    Array a(size);

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]
           && a.extent(2) == size[2]);

    index_type lb(0, 0, 0), ub;
    ub = size;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }

  //
  // Constructor: extents and initial value
  //
  {
    typedef Array<3> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type size(10, 20, 30);
    const double value = 42;
    Array a(size, value);

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]
           && a.extent(2) == size[2]);

    index_type lb(0, 0, 0), ub;
    ub = size;
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));

    for (Array::const_iterator i = a.begin(); i != a.end(); ++i) {
      assert(*i == value);
    }
  }

  //
  // Constructor: ranges
  //
  {
    typedef Array<1> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type lb, ub;
    lb = -4;
    ub = 6;
    index_type extents;
    extents = ub - lb;

    Array a(range_type(lb, ub));

    assert(a.size() == computeProduct(extents));

    assert(a.extents() == extents);
    assert(a.extent(0) == extents[0]);

    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<2> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type lb(1, 2), ub(10, 20);
    index_type size;
    size = ub - lb;
    Array a(range_type(lb, ub));

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]);

    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }
  {
    typedef Array<3> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type lb(1, 2, 3), ub(10, 20, 30);
    index_type size;
    size = ub - lb;
    Array a(range_type(lb, ub));

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]
           && a.extent(2) == size[2]);

    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));
  }

  //
  // Constructor: ranges and initial value
  //
  {
    typedef Array<3> Array;
    typedef Array::size_type size_type;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;

    index_type lb(1, 2, 3), ub(10, 20, 30);
    index_type size;
    size = ub - lb;
    const double value = 42;
    Array a(range_type(lb, ub), value);

    assert(a.size() == computeProduct(size));

    assert(a.extents() == size);
    assert(a.extent(0) == size[0] && a.extent(1) == size[1]
           && a.extent(2) == size[2]);

    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert(a.ranges() == range_type(lb, ub));

    assert(! a.empty());

    assert(a.getMemoryUsage() ==
           size_type(sizeof(Array) + sizeof(double) * a.size()));

    for (Array::const_iterator i = a.begin(); i != a.end(); ++i) {
      assert(*i == value);
    }
    for (int i = a.lbounds()[0]; i != a.ubounds()[0]; ++i) {
      for (int j = a.lbounds()[1]; j != a.ubounds()[1]; ++j) {
        for (int k = a.lbounds()[2]; k != a.ubounds()[2]; ++k) {
          assert(a(i, j , k) == value);
        }
      }
    }
  }

  //
  // Copy constructor.
  //
  {
    typedef Array<3, float>::index_type index_type;
    Array<3, float> a(index_type(10, 20, 30), 42.f);
    Array<3, float> b(a);
    assert(b == a);

    Array<3, float, false> r(a);
    Array<3, float> c(r);
    assert(r == a);

    Array<3, double> d(a);
    assert(d == a);
  }

  //
  // Range constructor.
  //
  {
    const int size = 24;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
      vec[i] = i;
    }
    typedef Array<3> Array;
    typedef Array::index_type index_type;
    Array a(index_type(2, 3, 4), vec.begin(), vec.end());

    // Accessors
    assert(a.size() == size);
    assert(a.extents() == index_type(2, 3, 4));
    assert(a.extent(0) == 2 && a.extent(1) == 3 &&
           a.extent(2) == 4);
    assert(! a.empty());
    assert(a.begin());
    assert(a.end() - a.begin() == size);

    // Indexing
    {
      index_type mi;
      int i = 0;
      for (mi[2] = 0; mi[2] < 4; ++mi[2]) {
        for (mi[1] = 0; mi[1] < 3; ++mi[1]) {
          for (mi[0] = 0; mi[0] < 2; ++mi[0]) {
            assert(a[i] == i);
            assert(a(mi) == vec[i++]);
            assert(a(mi[0], mi[1], mi[2]) == a(mi));
          }
        }
      }
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
    //typedef Array<3> Array;
    typedef Array<3>::index_type index_type;
    Array<3> a(index_type(2, 3, 4), vec.begin(), vec.end());

    // Assignment to Array.
    {
      Array<3> b(a.extents());
      b = a;
      assert(a == b);
    }
    {
      Array<3> b;
      b = a;
      assert(a == b);
    }
    // Assignment to value.
    {
      Array<3> b(index_type(2, 3, 5));
      b = 7;
      for (Array<3>::const_iterator i = b.begin();
           i != b.end(); ++i) {
        assert(*i == 7);
      }
    }
    // Assignment to array of different type.
    {
      Array<3, float> x(index_type(2, 3, 4), 1.f);
      Array<3> y;
      y = x;
      assert(y == x);
    }
  }

  //
  // Mathematical functions.
  //
  {
    Array<2> a(1, 1);
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
  // Conversions between multi-indices and container indices.
  //
  {
    typedef Array<2> Array;
    typedef Array::index_type index_type;
    typedef Array::range_type range_type;
    range_type range(10, 20, 20, 30);
    Array x(range);
    assert(x.index(index_type(10, 20)) == 0);
    // CONTINUE HERE
  }

  //
  // Swap.
  //
  {
    typedef Array<2> Array;
    typedef Array::index_type index_type;
    Array a(index_type(10, 10), 1.0);
    Array ac = a;
    assert(a == ac);
    Array b(index_type(20, 20), 2.0);
    Array bc = b;
    assert(b == bc);
    a.swap(b);
    assert(a == bc && b == ac);
  }
  //
  // I/O
  //
  {
    typedef Array<3> Array;
    typedef Array::index_type index_type;

    const index_type extents(2, 3, 4);
    Array a(extents);
    const int size = a.size();
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

// -*- C++ -*-

#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexExtentsIterator.h"
#include "stlib/container/MultiIndexRangeIterator.h"

#include <sstream>
#include <vector>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<std::size_t _N>
void
setToSum(container::SimpleMultiArray<int, _N>* a)
{
  typedef container::SimpleMultiIndexExtentsIterator<_N> Iterator;
  const Iterator end = Iterator::end(a->extents());
  for (Iterator i = Iterator::begin(a->extents()); i != end; ++i) {
    (*a)(*i) = stlib::ext::sum(*i);
  }
}

void
foo(double* data, const std::size_t extents[3])
{
  typedef container::SimpleMultiArrayRef<double, 3> SimpleMultiArrayRef;
  typedef SimpleMultiArrayRef::IndexList IndexList;
  SimpleMultiArrayRef a(data, ext::copy_array<IndexList>(extents));
}

int
main()
{
  // 1-D.
  {
    typedef int T;
    typedef container::SimpleMultiArray<T, 1> SimpleMultiArray;
    typedef SimpleMultiArray::pointer pointer;
    typedef SimpleMultiArray::iterator iterator;
    typedef SimpleMultiArray::IndexList IndexList;
    typedef SimpleMultiArray::Index Index;

    static_assert(SimpleMultiArray::Dimension == 1, "Bad dimension.");

    const std::size_t size = 7;
    const IndexList extents = {{size}};

    {
      SimpleMultiArray x(extents);
      assert(x.extents() == extents);
    }
    {
      SimpleMultiArray x(extents, 2);
      assert(x[0] == 2);
    }

    SimpleMultiArray x(extents);

    // Container accessors.
    assert(! x.empty());
    assert(x.size() == size);
    assert(x.max_size() == size);

    assert(x.begin() == x.data());
    assert(x.end() == x.data() + size);
    std::reverse_iterator<iterator> rbegin(iterator(x.data() + size));
    assert(x.rbegin() == rbegin);
    std::reverse_iterator<iterator> rend(iterator(x.data() + 0));
    assert(x.rend() == rend);

    for (std::size_t n = 0; n != size; ++n) {
      x[n] = n;
    }
    for (std::size_t n = 0; n != size; ++n) {
      assert(x[n] == int(n));
    }

    // SimpleMultiArray indexing.
    assert(x.extents() == extents);
    for (std::size_t n = 0; n != size; ++n) {
      IndexList i;
      i[0] = n;
      assert(x(i) == x.data()[n]);
    }
    {
      const SimpleMultiArray& y = x;
      for (std::size_t n = 0; n != size; ++n) {
        IndexList i;
        i[0] = n;
        assert(y(i) == y.data()[n]);
      }
    }
    assert(x.extents()[0] == size);
    assert(x.strides()[0] == 1);

    // Copy constructor.
    {
      SimpleMultiArray y(x);
      assert(y == x);
    }
    // Assignment operator.
    {
      SimpleMultiArray y = SimpleMultiArray(IndexList{{size}});
      y = x;
      assert(y == x);
    }
    {
      // Copy constructor for different types.
      const T data[size] = {0, 1, 2, 3, 4, 5, 6};
      container::SimpleMultiArrayConstRef<int, 1>
        y(data, IndexList{{size}});
      SimpleMultiArray z(y);
      assert(z == y);
      // Assignment operator for different types.
      x = y;
      assert(x == y);
    }

    // File I/O.
    {
      std::ostringstream out;
      out << x;
      std::istringstream in(out.str().c_str());
      SimpleMultiArray y;
      in >> y;
      assert(x == y);
    }

    // Rebuild
    {
      SimpleMultiArray y = x;
      pointer data = y.data();
      y.rebuild(y.extents());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.extents() + Index(1));
      assert(y != x);
      assert(y.extents() == x.extents() + Index(1));
    }
  }

  // Example code from the documentation.
  {
    // Array Examples
    typedef container::SimpleMultiArray<double, 2> SimpleMultiArray;
    typedef SimpleMultiArray::IndexList IndexList;
    IndexList extents = {{3, 4}};
    SimpleMultiArray a(extents);

    assert(! a.empty());
    assert(a.size() == 12);
    assert(a.max_size() == 12);
    std::fill(a.begin(), a.end(), 0);
    for (std::size_t i = 0; i != a.size(); ++i) {
      a[i] = i;
    }

    typedef SimpleMultiArray::IndexList IndexList;
    IndexList i;
    for (i[0] = 0; i[0] != 3; ++i[0]) {
      for (i[1] = 0; i[1] != 4; ++i[1]) {
        a(i) = stlib::ext::sum(i);
      }
    }
  }
  // The multidimensional array as a random access container
  {
    typedef container::SimpleMultiArray<double, 2> SimpleMultiArray;
    typedef SimpleMultiArray::IndexList IndexList;
    IndexList extents = {{3, 4}};
    SimpleMultiArray a(extents, 1.);

    typedef SimpleMultiArray::value_type value_type;
    value_type s = 0;
    for (std::size_t i = 0; i != a.size(); ++i) {
      s += a[i];
    }
    assert(s == sum(a));
  }
  {
    typedef container::SimpleMultiArray<double, 1> SimpleMultiArray;
    typedef SimpleMultiArray::IndexList IndexList;
    typedef SimpleMultiArray::size_type size_type;
    const IndexList extents = {{11}};
    SimpleMultiArray a(extents);
    for (size_type i = 0; i != a.size(); ++i) {
      a[i] = i;
    }

    std::vector<double> buffer(a.size());
    std::copy(a.begin(), a.end(), buffer.begin());

    SimpleMultiArray b(a.extents());
    assert(a.size() == b.size());
    std::copy(a.rbegin(), a.rend(), b.begin());

    std::fill(a.begin(), a.end(), 1);
    a.fill(1);
  }
  // Indexing operations.
  {
    typedef container::SimpleMultiArray<int, 3> SimpleMultiArray;
    typedef SimpleMultiArray::IndexList IndexList;
    const IndexList extents = {{11, 11, 11}};
    SimpleMultiArray a(extents);
    {
      IndexList i;
      for (i[0] = 0; i[0] != a.extents()[0]; ++i[0]) {
        for (i[1] = 0; i[1] != a.extents()[1]; ++i[1]) {
          for (i[2] = 0; i[2] != a.extents()[2]; ++i[2]) {
            a(i) = stlib::ext::product(i);
          }
        }
      }
    }
    typedef SimpleMultiArray::Index Index;
    for (Index i = 0; i != a.extents()[0]; ++i) {
      for (Index j = 0; j != a.extents()[1]; ++j) {
        for (Index k = 0; k != a.extents()[2]; ++k) {
          a(i, j, k) = i * j * k;
        }
      }
    }
  }
  // setToSum()
  {
    typedef container::SimpleMultiArray<int, 2> SimpleMultiArray;
    typedef SimpleMultiArray::IndexList IndexList;
    const IndexList extents = {{10, 20}};
    SimpleMultiArray a(extents);
    setToSum(&a);
  }
  // Multidimensional Array References
  {
    double data[] = {0};
    const std::size_t extents[3] = {1, 1, 1};
    foo(data, extents);
  }

  return 0;
}

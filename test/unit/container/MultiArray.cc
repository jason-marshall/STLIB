// -*- C++ -*-

#include "stlib/container/MultiArray.h"

#include <sstream>
#include <vector>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<std::size_t _N>
void
setToSum(container::MultiArray<int, _N>* a)
{
  typedef container::MultiIndexRangeIterator<_N> Iterator;
  const Iterator end = Iterator::end(a->range());
  for (Iterator i = Iterator::begin(a->range()); i != end; ++i) {
    (*a)(*i) = stlib::ext::sum(*i);
  }
}

template<typename _T, std::size_t _N>
void
laplacianAverage(const container::MultiArray<_T, _N>& a,
                 container::MultiArray<_T, _N>* b)
{
  assert(a.range() == b->range());
  typedef container::MultiIndexRangeIterator<_N> Iterator;
  typedef typename Iterator::IndexList IndexList;
  typedef typename Iterator::Index Index;
  typedef typename container::MultiArray<_T, _N>::size_type size_type;

  // Get the boundary values by copying the entire array.
  *b = a;

  // Skip if there are no interior points.
  if (stlib::ext::min(a.extents()) <= 2) {
    return;
  }

  // The range for the interior elements.
  const container::MultiIndexRange<_N>
    range(a.extents() - size_type(2), a.bases() + Index(1));
  // Compute the interior values.
  _T s;
  IndexList index;
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    s = 0;
    for (std::size_t n = 0; n != _N; ++n) {
      index = *i;
      index[n] -= 1;
      s += a(index);
      index[n] += 2;
      s += a(index);
    }
    (*b)(*i) = s / _T(2 * _N);
  }
}

void
foo(double* data, const std::size_t extents[3], const int bases[3])
{
  typedef container::MultiArrayRef<double, 3> MultiArrayRef;
  typedef MultiArrayRef::SizeList SizeList;
  typedef MultiArrayRef::IndexList IndexList;
  MultiArrayRef a(data, ext::copy_array<SizeList>(extents),
                  ext::copy_array<IndexList>(bases));
}

template<typename _T, std::size_t _N>
container::MultiArrayView<_T, _N>
interior(container::MultiArray<_T, _N>& a)
{
  typedef typename container::MultiArray<_T, _N>::Range Range;
  typedef typename container::MultiArray<_T, _N>::Index Index;
  typedef typename container::MultiArray<_T, _N>::size_type size_type;
  assert(stlib::ext::min(a.extents()) > 2);
  return a.view(Range(a.extents() - size_type(2), a.bases() + Index(1)));
}

int
main()
{
  // 1-D.
  {
    typedef int T;
    typedef container::MultiArray<T, 1> MultiArray;
    typedef MultiArray::pointer pointer;
    typedef MultiArray::iterator iterator;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::Storage Storage;
    typedef MultiArray::Index Index;
    typedef MultiArray::Range Range;
    typedef MultiArray::ConstView ConstView;
    typedef MultiArray::View View;

    static_assert(MultiArray::Dimension == 1, "Bad dimension.");

    const std::size_t size = 7;
    const SizeList extents = {{size}};
    const IndexList bases = {{0}};
    const Storage storage = container::ColumnMajor();

    {
      MultiArray x(extents);
      assert(x[0] == 0);
    }
    {
      MultiArray x(extents, 2);
      assert(x[0] == 2);
    }
    {
      MultiArray x(extents, storage);
      assert(x[0] == 0);
    }
    {
      MultiArray x(extents, storage, 3);
      assert(x[0] == 3);
    }
    {
      MultiArray x(extents, bases);
      assert(x[0] == 0);
    }
    {
      MultiArray x(extents, bases, 5);
      assert(x[0] == 5);
    }
    {
      MultiArray x(extents, bases, storage);
      assert(x[0] == 0);
    }
    {
      MultiArray x(extents, bases, storage, 7);
      assert(x[0] == 7);
    }

    MultiArray x(extents);

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

    // MultiArray indexing.
    assert(x.extents() == extents);
    assert(x.bases() == bases);
    assert(x.range() == Range(extents, bases));
    for (std::size_t n = 0; n != size; ++n) {
      IndexList i;
      i[0] = n;
      assert(x(i) == x.data()[n]);
    }
    {
      const MultiArray& y = x;
      for (std::size_t n = 0; n != size; ++n) {
        IndexList i;
        i[0] = n;
        assert(y(i) == y.data()[n]);
      }
    }
    assert(x.extents()[0] == size);
    assert(x.strides()[0] == 1);
    assert(x.bases()[0] == 0);

    // Copy constructor.
    {
      MultiArray y(x);
      assert(y == x);
    }
    // Assignment operator.
    {
      MultiArray y = MultiArray(SizeList{{size}});
      y = x;
      assert(y == x);
    }
    {
      // Copy constructor for different types.
      const T data[size] = {0, 1, 2, 3, 4, 5, 6};
      container::MultiArrayConstRef<int, 1>
        y(data, SizeList{{size}});
      MultiArray z(y);
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
      MultiArray y;
      in >> y;
      assert(x == y);
    }

    // Const sub-array.
    {
      const MultiArray& y = x;
      ConstView v = y.view(Range(extents, bases));
      assert(v == y);
    }
    // Sub-array.
    {
      View v = x.view(Range(extents, bases));
      assert(v == x);
      v(bases) = 33;
      assert(v(bases) == 33);
    }

    // Rebuild
    {
      MultiArray y = x;
      pointer data = y.data();
      y.rebuild(y.extents());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.extents(), y.bases());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.extents(), y.bases(), y.storage());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.extents(), y.bases() + Index(1));
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.extents() + std::size_t(1));
      assert(y != x);
      assert(y.extents() == x.extents() + std::size_t(1));
    }
  }

  // Example code from the documentation.
  {
    // Array Examples
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    SizeList extents = {{3, 4}};
    MultiArray a(extents);

    assert(! a.empty());
    assert(a.size() == 12);
    assert(a.max_size() == 12);
    std::fill(a.begin(), a.end(), 0);
    for (std::size_t i = 0; i != a.size(); ++i) {
      a[i] = i;
    }

    typedef MultiArray::IndexList IndexList;
    IndexList i;
    for (i[0] = 0; i[0] != 3; ++i[0]) {
      for (i[1] = 0; i[1] != 4; ++i[1]) {
        a(i) = stlib::ext::sum(i);
      }
    }
  }
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::Index Index;
    typedef MultiArray::IndexList IndexList;
    const SizeList extents = {{3, 4}};
    const IndexList bases = {{1, 1}};
    MultiArray b(extents, bases);

    const IndexList lower = b.bases();
    IndexList upper = lower;
    upper += ext::ConvertArray<Index>::convert(b.extents());
    IndexList i;
    for (i[0] = lower[0]; i[0] != upper[0]; ++i[0]) {
      for (i[1] = lower[0]; i[1] != upper[1]; ++i[1]) {
        b(i) = stlib::ext::sum(i);
      }
    }
  }
  // The Multidimensional Array as a Random Access Container
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    SizeList extents = {{3, 4}};
    MultiArray a(extents, 1.);

    typedef MultiArray::value_type value_type;
    value_type s = 0;
    for (std::size_t i = 0; i != a.size(); ++i) {
      s += a[i];
    }
    assert(s == sum(a));
  }
  {
    typedef container::MultiArray<double, 1> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::size_type size_type;
    const SizeList extents = {{11}};
    const IndexList bases = {{ -5}};
    MultiArray a(extents, bases);
    for (size_type i = 0; i != a.size(); ++i) {
      a[i] = i;
    }

    std::vector<double> buffer(a.size());
    std::copy(a.begin(), a.end(), buffer.begin());

    MultiArray b(a.extents());
    assert(a.size() == b.size());
    std::copy(a.rbegin(), a.rend(), b.begin());

    std::fill(a.begin(), a.end(), 1);
    a.fill(1);
  }
  // Indexing operations.
  {
    typedef container::MultiArray<int, 3> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::Index Index;
    typedef MultiArray::IndexList IndexList;
    const SizeList extents = {{11, 11, 11}};
    const IndexList bases = {{ -5, -5, -5}};
    MultiArray a(extents, bases);
    const IndexList lower = a.bases();
    IndexList upper = a.bases();
    upper += ext::ConvertArray<Index>::convert(a.extents());
    {
      IndexList i;
      for (i[0] = lower[0]; i[0] != upper[0]; ++i[0]) {
        for (i[1] = lower[1]; i[1] != upper[1]; ++i[1]) {
          for (i[2] = lower[2]; i[2] != upper[2]; ++i[2]) {
            a(i) = stlib::ext::product(i);
          }
        }
      }
    }
    typedef MultiArray::Index Index;
    for (Index i = lower[0]; i != upper[0]; ++i) {
      for (Index j = lower[1]; j != upper[1]; ++j) {
        for (Index k = lower[2]; k != upper[2]; ++k) {
          a(i, j, k) = i * j * k;
        }
      }
    }
  }
  // Index Ranges and Their Iterators
  {
    typedef container::MultiIndexRange<2> Range;
    typedef Range::SizeList SizeList;
    {
      const SizeList extents = {{3, 4}};
      Range range(extents);
    }
    typedef Range::IndexList IndexList;
    {
      const SizeList extents = {{3, 4}};
      const IndexList bases = {{1, 1}};
      Range range(extents, bases);
    }
    {
      const SizeList extents = {{3, 4}};
      const IndexList bases = {{1, 1}};
      const IndexList steps = {{2, 3}};
      Range range(extents, bases, steps);
    }
  }
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::Range Range;
    const SizeList extents = {{10, 20}};
    MultiArray a(extents);
    Range range = a.range();
    assert(range.extents() == a.extents());
    assert(range.bases() == a.bases());
    assert(range.steps() == ext::filled_array<IndexList>(1));
  }
  {
    typedef container::MultiArray<int, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    const SizeList extents = {{10, 20}};
    MultiArray a(extents);
    setToSum(&a);
  }
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    const SizeList extents = {{10, 20}};
    MultiArray a(extents, 1.), b(extents);
    laplacianAverage(a, &b);
  }
  // Multidimensional Array References
  {
    double data[] = {0};
    const std::size_t extents[3] = {1, 1, 1};
    const int bases[3] = {0, 0, 0};
    foo(data, extents, bases);
  }
  // Multidimensional Array Views
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::Range Range;
    typedef MultiArray::View View;
    const SizeList extents = {{12, 12}};
    const IndexList bases = {{ -1, -1}};
    MultiArray a(extents, bases);
    const SizeList viewExtents = {{10, 10}};
    View interior = a.view(Range(viewExtents));
  }
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::View View;
    const SizeList extents = {{12, 12}};
    const IndexList bases = {{ -1, -1}};
    MultiArray a(extents, bases);
    View b = interior(a);
  }
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    const SizeList extents = {{12, 12}};
    const IndexList bases = {{ -1, -1}};
    MultiArray a(extents, bases);
    container::MultiArrayView<double, 2> b = interior(a);
    typedef container::MultiArrayView<double, 2>::iterator iterator;
    const iterator end = b.end();
    for (iterator i = b.begin(); i != end; ++i) {
      *i = 0;
    }
    for (iterator i = b.begin(); i != end; ++i) {
      assert(*i == b(i.indexList()));
    }
  }

  return 0;
}

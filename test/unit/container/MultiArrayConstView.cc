// -*- C++ -*-

#include "stlib/container/MultiArrayConstView.h"

using namespace stlib;

void
test1(const int* data,
      const container::MultiArrayConstView<int, 1>::SizeList& extents,
      const container::MultiArrayConstView<int, 1>::IndexList& bases,
      const container::MultiArrayConstView<int, 1>::Storage& storage,
      const container::MultiArrayConstView<int, 1>::IndexList& strides)
{
  const std::size_t Dimension = 1;
  typedef container::MultiArrayConstView<int, Dimension> MultiArrayConstView;
  typedef MultiArrayConstView::const_iterator const_iterator;
  typedef MultiArrayConstView::const_reverse_iterator const_reverse_iterator;
  typedef MultiArrayConstView::size_type size_type;
  typedef MultiArrayConstView::Index Index;
  typedef MultiArrayConstView::IndexList IndexList;
  typedef MultiArrayConstView::Range Range;
  typedef MultiArrayConstView::ConstView ConstView;

  static_assert(MultiArrayConstView::Dimension == Dimension, "Bad dimension.");
  const size_type size = extents[0];

  MultiArrayConstView x(data, extents, bases, storage, strides);

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);
  assert(std::distance(x.begin(), x.end()) == Index(size));
  assert(std::distance(x.rbegin(), x.rend()) == Index(size));
  for (std::size_t n = 0; n != size; ++n) {
    assert(x.begin()[n] == data[n]);
  }
  for (const_iterator i = x.begin(); i != x.end(); ++i) {
    assert(*i == data[i.rank()]);
  }
  {
    size_type n = size;
    for (const_reverse_iterator i = x.rbegin(); i != x.rend(); ++i) {
      assert(*i == data[--n]);
    }
  }

  // MultiArray indexing.
  for (size_type n = 0; n != size; ++n) {
    IndexList i;
    i[0] = bases[0] + n;
    assert(x(i) == data[n]);
    assert(x(i[0]) == data[n]);
  }
  assert(x.data() == data);
  assert(x.extents() == extents);
  assert(x.bases() == bases);
  assert(x.strides() == strides);

  // Sub-array.
  {
    ConstView v = x.view(Range(extents, bases));
    assert(v == x);
  }

  // Mathematical functions.
  assert(sum(x) == std::accumulate(x.begin(), x.end(), 0));
  assert(product(x) ==
         std::accumulate(x.begin(), x.end(), 1, std::multiplies<int>()));
  assert(min(x) == *std::min_element(x.begin(), x.end()));
  assert(max(x) == *std::max_element(x.begin(), x.end()));
}

int
main()
{
  // 1-D.
  {
    const std::size_t Dimension = 1;
    typedef int T;
    typedef container::MultiArrayConstView<T, Dimension> MultiArrayConstView;
    typedef MultiArrayConstView::size_type size_type;
    typedef MultiArrayConstView::SizeList SizeList;
    typedef MultiArrayConstView::IndexList IndexList;

    const T data[] = {0, 1, 2, 3, 4, 5, 6};
    const size_type size = sizeof(data) / sizeof(T);

    test1(data, SizeList{{size}}, IndexList{{0}},
          container::RowMajor(), IndexList{{1}});
    test1(data, SizeList{{size}}, IndexList{{5}},
          container::RowMajor(), IndexList{{1}});
  }

  return 0;
}

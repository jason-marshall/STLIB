// -*- C++ -*-

#include "stlib/container/MultiArrayView.h"

using namespace stlib;

void
test1(int* data,
      const container::MultiArrayView<int, 1>::SizeList& extents,
      const container::MultiArrayView<int, 1>::IndexList& bases,
      const container::MultiArrayView<int, 1>::Storage& storage,
      const container::MultiArrayView<int, 1>::IndexList& strides)
{
  const std::size_t Dimension = 1;
  typedef container::MultiArrayView<int, Dimension> MultiArrayView;
  typedef MultiArrayView::const_iterator const_iterator;
  typedef MultiArrayView::size_type size_type;
  typedef MultiArrayView::Index Index;
  typedef MultiArrayView::IndexList IndexList;
  typedef MultiArrayView::Range Range;
  typedef MultiArrayView::ConstView ConstView;
  typedef MultiArrayView::View View;

  static_assert(MultiArrayView::Dimension == Dimension, "Bad dimension.");
  const size_type size = extents[0];

  MultiArrayView x(data, extents, bases, storage, strides);

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);

  // MultiArray indexing.
  for (std::size_t n = 0; n != size; ++n) {
    IndexList i;
    i[0] = bases[0] + Index(n);
    x(i) = data[n];
    assert(x(i) == data[n]);
    assert(x(i[0]) == data[n]);
  }
  assert(x.data() == data);
  assert(x.extents() == extents);
  assert(x.bases() == bases);
  assert(x.strides() == strides);

  // Copy constructor.
  {
    MultiArrayView y(x);
    //assert(y == x);
  }
  {
    container::MultiArrayView<int, 1> y(x);
    assert(y.extents() == x.extents());
    //assert(y == x);
  }

  // Const sub-array.
  {
    const MultiArrayView& y = x;
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


  // Mathematical functions.
  assert(sum(x) == std::accumulate(x.begin(), x.end(), 0));
  assert(product(x) ==
         std::accumulate(x.begin(), x.end(), 1, std::multiplies<int>()));
  assert(min(x) == *std::min_element(x.begin(), x.end()));
  assert(max(x) == *std::max_element(x.begin(), x.end()));

  // Assignment operators with scalar operand.
  {
    // Fill.
    x.fill(1);
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 1);
    }
  }
  {
    // +=
    x += 1;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 2);
    }
  }
  {
    // -=
    x -= 1;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 1);
    }
  }
  {
    // *=
    x *= 20;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 20);
    }
  }
  {
    // /=
    x /= 4;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 5);
    }
  }
  {
    // %=
    x %= 3;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 2);
    }
  }
  {
    // <<=
    x <<= 1;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 4);
    }
  }
  {
    // >>=
    x >>= 1;
    const const_iterator end = x.end();
    for (const_iterator i = x.begin(); i != end; ++i) {
      assert(*i == 2);
    }
  }
}

int
main()
{
  // 1-D.
  {
    const std::size_t Dimension = 1;
    typedef int T;
    typedef container::MultiArrayView<T, Dimension> MultiArrayView;
    typedef MultiArrayView::size_type size_type;
    typedef MultiArrayView::SizeList SizeList;
    typedef MultiArrayView::IndexList IndexList;

    const size_type size = 7;
    T data[size];
    test1(data, SizeList{{size}}, IndexList{{0}},
          container::RowMajor(), IndexList{{1}});
    test1(data, SizeList{{size}}, IndexList{{5}},
          container::RowMajor(), IndexList{{1}});
  }

  return 0;
}

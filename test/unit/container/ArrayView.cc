// -*- C++ -*-

#include "stlib/container/ArrayView.h"

using namespace stlib;

void
test1(int* data,
      const container::ArrayView<int>::size_type size,
      const container::ArrayView<int>::Index base,
      const container::ArrayView<int>::Index stride)
{
  typedef container::ArrayView<int> ArrayView;
  typedef ArrayView::const_iterator const_iterator;
  typedef ArrayView::Index Index;
  typedef ArrayView::Range Range;
  typedef ArrayView::ConstView ConstView;
  typedef ArrayView::View View;

  ArrayView x(data, size, base, stride);

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);

  // Array indexing.
  for (std::size_t n = 0; n != size; ++n) {
    Index i = Index(n) + x.base();
    x(i) = data[n];
    assert(x(i) == data[n]);
  }
  assert(x.data() == data);
  assert(x.base() == base);
  assert(x.stride() == stride);

  // Copy constructor.
  {
    ArrayView y(x);
    //assert(y == x);
  }
  {
    container::ArrayView<int> y(x);
    assert(y.size() == x.size());
    //assert(y == x);
  }

  // Const sub-array.
  {
    const ArrayView& y = x;
    ConstView v = y.view(Range(size, base));
    assert(v == y);
  }
  // Sub-array.
  {
    View v = x.view(Range(size, base));
    assert(v == x);
    v(base) = 33;
    assert(v(base) == 33);
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
  {
    typedef int T;
    typedef container::ArrayView<T> ArrayView;
    typedef ArrayView::size_type size_type;

    const size_type size = 7;
    T data[size];
    test1(data, size, 0, 1);
    test1(data, size, 5, 1);
  }

  return 0;
}

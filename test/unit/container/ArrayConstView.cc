// -*- C++ -*-

#include "stlib/container/ArrayConstView.h"

using namespace stlib;

void
test1(const int* data,
      const container::ArrayConstView<int>::size_type size,
      const container::ArrayConstView<int>::Index base,
      const container::ArrayConstView<int>::Index stride)
{
  typedef container::ArrayConstView<int> ArrayConstView;
  typedef ArrayConstView::const_iterator const_iterator;
  typedef ArrayConstView::const_reverse_iterator const_reverse_iterator;
  typedef ArrayConstView::size_type size_type;
  typedef ArrayConstView::Index Index;
  typedef ArrayConstView::Range Range;
  typedef ArrayConstView::ConstView ConstView;

  ArrayConstView x(data, size, base, stride);

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

  // Array indexing.
  for (size_type n = 0; n != size; ++n) {
    Index i;
    i = base + n;
    assert(x(i) == data[n]);
  }
  assert(x.data() == data);
  assert(x.size() == size);
  assert(x.base() == base);
  assert(x.stride() == stride);

  // Sub-array.
  {
    ConstView v = x.view(Range(size, base));
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
  {
    typedef int T;
    typedef container::ArrayConstView<T> ArrayConstView;
    typedef ArrayConstView::size_type size_type;

    const T data[] = {0, 1, 2, 3, 4, 5, 6};
    const size_type size = sizeof(data) / sizeof(T);

    test1(data, size, 0, 1);
    test1(data, size, 5, 1);
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/container/ArrayConstRef.h"

using namespace stlib;

int
main()
{
  {
    typedef int T;
    typedef container::ArrayConstRef<T> ArrayConstRef;
    typedef ArrayConstRef::const_iterator const_iterator;
    typedef ArrayConstRef::Index Index;
    typedef ArrayConstRef::Range Range;
    typedef ArrayConstRef::ConstView ConstView;

    const T data[] = {0, 1, 2, 3, 4, 5, 6};
    const std::size_t size = sizeof(data) / sizeof(T);

    // Size constructor.
    {
      const Index base = 0;
      ArrayConstRef x(data, size);

      // Container accessors.
      assert(! x.empty());
      assert(x.size() == size);
      assert(x.max_size() == size);
      assert(x.begin() == data);
      assert(x.end() == data + size);
      std::reverse_iterator<const_iterator> rbegin(const_iterator(data + size));
      assert(x.rbegin() == rbegin);
      std::reverse_iterator<const_iterator> rend(const_iterator(data + 0));
      assert(x.rend() == rend);
      for (std::size_t n = 0; n != size; ++n) {
        assert(x[n] == data[n]);
      }

      // Array indexing.
      for (std::size_t n = 0; n != size; ++n) {
        Index i = n;
        assert(x(i) == data[n]);
      }
      assert(x.data() == data);
      assert(x.size() == size);
      assert(x.base() == 0);
      assert(x.stride() == 1);

      // File I/O.
      std::cout << x << '\n';
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

    // Extents and bases constructor.
    {
      const Index base = 5;
      ArrayConstRef x(data, size, base);

      // Container accessors.
      assert(! x.empty());
      assert(x.size() == size);
      assert(x.max_size() == size);
      assert(x.begin()  == data);
      assert(x.end() == data + size);
      std::reverse_iterator<const_iterator> rbegin(const_iterator(data + size));
      assert(x.rbegin() == rbegin);
      std::reverse_iterator<const_iterator> rend(const_iterator(data + 0));
      assert(x.rend() == rend);
      for (std::size_t n = 0; n != size; ++n) {
        assert(x[n] == data[n]);
      }

      // Array indexing.
      for (std::size_t n = 0; n != size; ++n) {
        Index i = base + n;
        assert(x(i) == data[n]);
      }
      assert(x.data() == data);
      assert(x.size() == size);
      assert(x.base() == base);
      assert(x.stride() == 1);

      // File I/O.
      std::cout << x << '\n';

      // Sub-array.
      {
        ConstView v = x.view(Range(size, base));
        assert(v == x);
      }

      // Rebuild
      {
        ArrayConstRef y(x);
        y.rebuild(x);
        assert(y == x);
      }
      {
        ArrayConstRef y = x;
        y.rebuild(x.base());
        assert(y == x);
        y.rebuild(x.base() + Index(1));
        assert(y == x);
      }
      {
        ArrayConstRef y = x;
        y.rebuild(x.data(), x.size(), x.base());
        assert(y == x);
      }
    }
  }

  return 0;
}

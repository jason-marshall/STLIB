// -*- C++ -*-

#include "stlib/container/Array.h"

using namespace stlib;

int
main()
{
  {
    typedef int T;
    typedef container::Array<T> Array;
    typedef Array::pointer pointer;
    typedef Array::iterator iterator;
    typedef Array::Index Index;
    typedef Array::Range Range;
    typedef Array::ConstView ConstView;
    typedef Array::View View;

    const std::size_t size = 7;
    const Index base = 0;
    Array x = Array(size);

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

    // Array indexing.
    assert(x.base() == base);
    assert(x.range() == Range(size, base));
    for (std::size_t n = 0; n != size; ++n) {
      Index i = n;
      assert(x(i) == x.data()[n]);
    }
    {
      const Array& y = x;
      for (std::size_t n = 0; n != size; ++n) {
        Index i = n;
        assert(y(i) == y.data()[n]);
      }
    }
    assert(x.size() == size);
    assert(x.stride() == 1);
    assert(x.base() == 0);

    // Copy constructor.
    {
      Array y(x);
      assert(y == x);
    }
    // Assignment operator.
    {
      Array y = Array(size);
      y = x;
      assert(y == x);
    }
    {
      // Copy constructor for different types.
      const T data[size] = {0, 1, 2, 3, 4, 5, 6};
      container::ArrayConstRef<int>	y(data, size);
      Array z(y);
      assert(z == y);
      // Assignment operator for different types.
      x = y;
      assert(x == y);
    }

    // File I/O.
    std::cout << x;

    // Const sub-array.
    {
      const Array& y = x;
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

    // Rebuild
    {
      Array y;
      y.rebuild(x);
      assert(y == x);
    }
    {
      Array y = x;
      pointer data = y.data();
      y.rebuild(y.size());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.size(), y.base());
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.size(), y.base() + Index(1));
      assert(y == x);
      assert(y.data() == data);
      y.rebuild(y.size() + std::size_t(1));
      assert(y != x);
      assert(y.data() != data);
      assert(y.size() == x.size() + std::size_t(1));
    }
  }

  return 0;
}

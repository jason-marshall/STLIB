// -*- C++ -*-

#include "stlib/container/ArrayRef.h"

using namespace stlib;

int
main()
{
  {
    typedef int T;
    typedef container::ArrayRef<T> ArrayRef;
    typedef ArrayRef::iterator iterator;
    typedef ArrayRef::const_iterator const_iterator;
    typedef ArrayRef::Index Index;
    typedef ArrayRef::Range Range;
    typedef ArrayRef::ConstView ConstView;
    typedef ArrayRef::View View;

    T data[7];
    const std::size_t size = sizeof(data) / sizeof(T);
    const Index base = 0;
    ArrayRef x(data, size);

    // Container accessors.
    assert(! x.empty());
    assert(x.size() == size);
    assert(x.max_size() == size);

    assert(x.begin() == data);
    assert(x.end() == data + size);
    std::reverse_iterator<iterator> rbegin(iterator(data + size));
    assert(x.rbegin() == rbegin);
    std::reverse_iterator<iterator> rend(iterator(data + 0));
    assert(x.rend() == rend);

    {
      const ArrayRef& y = x;
      assert(y.begin() == data);
      assert(y.end() == data + size);
      std::reverse_iterator<const_iterator> rbegin(const_iterator(data + size));
      assert(y.rbegin() == rbegin);
      std::reverse_iterator<const_iterator> rend(const_iterator(data + 0));
      assert(y.rend() == rend);
    }

    for (std::size_t n = 0; n != size; ++n) {
      x[n] = n;
    }
    for (std::size_t n = 0; n != size; ++n) {
      assert(data[n] == int(n));
    }

    // Array indexing.
    for (std::size_t n = 0; n != size; ++n) {
      Index i = n;
      assert(x(i) == data[n]);
    }
    assert(x.data() == data);
    assert(x.size() == size);
    assert(x.stride() == 1);
    assert(x.base() == 0);

    // Copy constructor.
    {
      ArrayRef y(x);
      assert(y == x);
      assert(y.begin() == x.begin());
      assert(y.end() == x.end());
    }
    {
      container::ArrayConstRef<int> y(x);
      assert(y.size() == x.size());
      assert(std::equal(x.begin(), x.end(), y.begin()));
      assert(y == x);
    }

    // Assignment operator.
    {
      const T d[size] = {0, -1, -2, -3, -4, -5, -6};
      container::ArrayConstRef<int> y(d, size);
      x = y;
      assert(x == y);
    }
    {
      T d[size];
      container::ArrayRef<int> y(d, size);
      y = x;
      assert(x == y);
    }

    // File I/O.
    std::cout << x;

    // Const sub-array.
    {
      const ArrayRef& y = x;
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
      ArrayRef y = x;
      y.rebuild(x);
      assert(y == x);
    }
    {
      ArrayRef y = x;
      y.rebuild(y.base());
      assert(y == x);
      y.rebuild(y.base() + Index(1));
      assert(y == x);
    }
    {
      ArrayRef y = x;
      y.rebuild(x.data(), x.size(), x.base());
      assert(y == x);
    }

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

  return 0;
}

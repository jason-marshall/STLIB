// -*- C++ -*-

#include "stlib/container/SimpleMultiArrayRef.h"

#include <sstream>

using namespace stlib;

int
main()
{
  // 1-D.
  {
    typedef int T;
    typedef container::SimpleMultiArrayRef<T, 1> SimpleMultiArrayRef;
    typedef SimpleMultiArrayRef::iterator iterator;
    typedef SimpleMultiArrayRef::const_iterator const_iterator;
    typedef SimpleMultiArrayRef::IndexList IndexList;

    static_assert(SimpleMultiArrayRef::Dimension == 1, "Bad dimension.");

    T data[7];
    const std::size_t size = sizeof(data) / sizeof(T);
    const IndexList extents = {{size}};
    SimpleMultiArrayRef x(data, extents);

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
      const SimpleMultiArrayRef& y = x;
      assert(y.begin() == data);
      assert(y.end() == data + size);
      std::reverse_iterator<const_iterator>
      rbegin(const_iterator(data + size));
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

    // MultiArray indexing.
    for (std::size_t n = 0; n != size; ++n) {
      IndexList i;
      i[0] = n;
      assert(x(i) == data[n]);
    }
    assert(x.data() == data);
    assert(x.extents()[0] == size);
    assert(x.strides()[0] == 1);

    // Copy constructor.
    {
      SimpleMultiArrayRef y(x);
      assert(y == x);
      assert(y.begin() == x.begin());
      assert(y.end() == x.end());
    }
    {
      container::SimpleMultiArrayConstRef<int, 1> y(x);
      assert(y.extents() == x.extents());
      assert(std::equal(x.begin(), x.end(), y.begin()));
      assert(y == x);
    }

    // Assignment operator.
    {
      const T d[size] = {0, -1, -2, -3, -4, -5, -6};
      container::SimpleMultiArrayConstRef<int, 1> y(d, IndexList{{size}});
      x = y;
      assert(x == y);
    }
    {
      T d[size];
      container::SimpleMultiArrayRef<int, 1> y(d, IndexList{{size}});
      y = x;
      assert(x == y);
    }

    // File I/O.
    {
      std::ostringstream out;
      out << x;
      std::istringstream in(out.str().c_str());
      T d[size];
      SimpleMultiArrayRef y(d, extents);
      in >> y;
      assert(x == y);
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

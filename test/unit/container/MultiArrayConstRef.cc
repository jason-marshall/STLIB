// -*- C++ -*-

#include "stlib/container/MultiArrayConstRef.h"

#include <sstream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  // 1-D.
  {
    typedef int T;
    typedef container::MultiArrayConstRef<T, 1> MultiArrayConstRef;
    typedef MultiArrayConstRef::const_iterator const_iterator;
    typedef MultiArrayConstRef::SizeList SizeList;
    typedef MultiArrayConstRef::IndexList IndexList;
    typedef MultiArrayConstRef::Index Index;
    typedef MultiArrayConstRef::Range Range;
    typedef MultiArrayConstRef::ConstView ConstView;

    static_assert(MultiArrayConstRef::Dimension == 1, "Bad dimension.");

    const T data[] = {0, 1, 2, 3, 4, 5, 6};
    const std::size_t size = sizeof(data) / sizeof(T);

    // Extent constructor.
    {
      const SizeList extents = {{size}};
      const IndexList bases = {{0}};
      MultiArrayConstRef x(data, extents);

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

      // MultiArray indexing.
      for (std::size_t n = 0; n != size; ++n) {
        IndexList i;
        i[0] = n;
        assert(x(i) == data[n]);
      }
      assert(x.data() == data);
      assert(x.extents()[0] == size);
      assert(x.bases()[0] == 0);
      assert(x.strides()[0] == 1);

      // File I/O.
      std::cout << x << '\n';
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

    // Extents and bases constructor.
    {
      const SizeList extents = {{size}};
      const IndexList bases = {{5}};
      MultiArrayConstRef x(data, extents, bases);

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

      // MultiArray indexing.
      for (std::size_t n = 0; n != size; ++n) {
        IndexList i;
        i[0] = bases[0] + n;
        assert(x(i) == data[n]);
      }
      assert(x.data() == data);
      assert(x.extents()[0] == size);
      assert(x.bases()[0] == bases[0]);
      assert(x.strides()[0] == 1);

      // File I/O.
      std::cout << x << '\n';
      {
        std::ostringstream file;
        write(x, file);
      }

      // Sub-array.
      {
        ConstView v = x.view(Range(extents, bases));
        assert(v == x);
      }

      // Rebuild
      {
        MultiArrayConstRef y = x;
        y.rebuild(y.extents());
        assert(y == x);
        y.rebuild(y.extents(), y.bases());
        assert(y == x);
        y.rebuild(y.extents(), y.bases(), y.storage());
        assert(y == x);
        y.rebuild(y.extents(), y.bases() + Index(1));
        assert(y == x);
      }
    }
  }

  return 0;
}

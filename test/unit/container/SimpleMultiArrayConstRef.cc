// -*- C++ -*-

#include "stlib/container/SimpleMultiArrayConstRef.h"

#include <sstream>

using namespace stlib;

int
main()
{
  // 1-D.
  {
    typedef int T;
    typedef container::SimpleMultiArrayConstRef<T, 1> SimpleMultiArrayConstRef;
    typedef SimpleMultiArrayConstRef::const_iterator const_iterator;
    typedef SimpleMultiArrayConstRef::IndexList IndexList;

    static_assert(SimpleMultiArrayConstRef::Dimension == 1, "Bad dimension.");

    const T data[] = {0, 1, 2, 3, 4, 5, 6};
    const std::size_t size = sizeof(data) / sizeof(T);

    // Extent constructor.
    {
      const IndexList extents = {{size}};
      SimpleMultiArrayConstRef x(data, extents);

      // Container accessors.
      assert(! x.empty());
      assert(x.size() == size);
      assert(x.max_size() == size);
      assert(x.begin() == data);
      assert(x.end() == data + size);
      std::reverse_iterator<const_iterator>
      rbegin(const_iterator(data + size));
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
      assert(x.strides()[0] == 1);

      // File I/O.
      std::cout << x << '\n';

      // Mathematical functions.
      assert(sum(x) == std::accumulate(x.begin(), x.end(), 0));
      assert(product(x) ==
             std::accumulate(x.begin(), x.end(), 1, std::multiplies<int>()));
      assert(min(x) == *std::min_element(x.begin(), x.end()));
      assert(max(x) == *std::max_element(x.begin(), x.end()));
    }
  }

  return 0;
}

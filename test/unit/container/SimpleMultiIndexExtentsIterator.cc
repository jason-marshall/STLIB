// -*- C++ -*-

#include "stlib/container/SimpleMultiIndexExtentsIterator.h"

using namespace stlib;

int
main()
{
  //--------------------------------------------------------------------------
  // 1-D.
  //--------------------------------------------------------------------------
  {
    typedef container::SimpleMultiIndexExtentsIterator<1> Iterator;
    typedef Iterator::IndexList IndexList;
    typedef Iterator::Index Index;
    typedef Iterator::size_type size_type;

    static_assert(Iterator::Dimension == 1, "Bad dimension.");

    const size_type size = 7;
    IndexList extents = {{size}};

    //
    // Range with extent constructor.
    //
    {
      // Beginning of the range.
      Iterator begin = Iterator::begin(extents);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(extents);
        i = begin;
        assert(i == begin);
      }
      {
        // Extents constructor.
        Iterator i = Iterator::begin(extents);
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(extents);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(extents);
        i = end;
        assert(i == end);
      }
      {
        // Extents constructor.
        Iterator i = Iterator::end(extents);
        assert(i == end);
      }

      assert(begin.extents() == extents);
      assert(end.extents() == extents);
      assert(*begin == IndexList{{0}});
      assert(*end == IndexList{{size}});

      {
        // Pre-increment.
        size_type count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        size_type count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        size_type count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        size_type count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        for (size_type i = 0; i != size; ++i) {
          assert(begin[i] == IndexList{{i}});
        }
      }
      {
        // +=
        Iterator i = begin;
        i += size;
        assert(i == end);
      }
      {
        // +
        assert(begin + size == end);
      }
      {
        // -=
        Iterator i = end;
        i -= size;
        assert(i == begin);
      }
      {
        // -
        assert(begin == end - size);
      }
      {
        // base
        assert(begin.base() == 0);
        assert(end.base() == Index(size));
      }

      {
        // Comparison.
        assert(begin == begin);
        assert(begin != end);
        assert(begin < end);
        assert(begin <= begin);
        assert(begin <= end);
        assert(end > begin);
        assert(end >= end);
        assert(end >= begin);
      }

      assert(std::distance(begin, end) == Index(size));
      assert(size + begin == end);
    }
  }

  //--------------------------------------------------------------------------
  // 2-D.
  //--------------------------------------------------------------------------
  {
    typedef container::SimpleMultiIndexExtentsIterator<2> Iterator;
    typedef Iterator::IndexList IndexList;
    typedef Iterator::Index Index;
    typedef Iterator::size_type size_type;

    static_assert(Iterator::Dimension == 2, "Bad dimension.");

    IndexList extents = {{2, 3}};
    const size_type size = 6;

    //
    // Range with extent constructor.
    //
    {
      // Beginning of the range.
      Iterator begin = Iterator::begin(extents);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(extents);
        i = begin;
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(extents);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(extents);
        i = end;
        assert(i == end);
      }

      assert(begin.extents() == extents);
      assert(end.extents() == extents);
      assert(*begin == (IndexList{{0, 0}}));
      assert(*end == (IndexList{{0, extents[1]}}));

      {
        // Pre-increment.
        size_type count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        size_type count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        size_type count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        size_type count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        size_type n = 0;
        for (size_type j = 0; j != extents[1]; ++j) {
          for (size_type i = 0; i != extents[0]; ++i) {
            assert(begin[n++] == (IndexList{{i, j}}));
          }
        }
      }
      {
        // +=
        Iterator i = begin;
        i += size;
        assert(i == end);
      }
      {
        // +
        assert(begin + size == end);
      }
      {
        // -=
        Iterator i = end;
        i -= size;
        assert(i == begin);
      }
      {
        // -
        assert(begin == end - size);
      }
      {
        // base
        assert(begin.base() == 0);
        assert(end.base() == Index(size));
      }

      {
        // Comparison.
        assert(begin == begin);
        assert(begin != end);
        assert(begin < end);
        assert(begin <= begin);
        assert(begin <= end);
        assert(end > begin);
        assert(end >= end);
        assert(end >= begin);
      }

      assert(std::distance(begin, end) == Index(size));
      assert(size + begin == end);
    }
  }

  return 0;
}

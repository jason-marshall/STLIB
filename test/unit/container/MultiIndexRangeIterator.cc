// -*- C++ -*-

#include "stlib/container/MultiIndexRangeIterator.h"

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  //--------------------------------------------------------------------------
  // 1-D.
  //--------------------------------------------------------------------------
  {
    typedef container::MultiIndexRangeIterator<1> Iterator;
    typedef Iterator::Range Range;
    typedef Iterator::SizeList SizeList;
    typedef Iterator::IndexList IndexList;
    typedef Iterator::Index Index;
    typedef Iterator::size_type size_type;

    static_assert(Iterator::Dimension == 1, "Bad dimension.");

    const size_type size = 7;
    SizeList extents = {{size}};

    //
    // Range with extent constructor.
    //
    {
      Range range(extents);

      // Beginning of the range.
      Iterator begin = Iterator::begin(range);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = begin;
        assert(i == begin);
      }
      {
        // Extents constructor.
        Iterator i = Iterator::begin(extents);
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(range);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = end;
        assert(i == end);
      }
      {
        // Extents constructor.
        Iterator i = Iterator::end(extents);
        assert(i == end);
      }

      assert(begin.range() == range);
      assert(end.range() == range);
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
          assert(begin[i] == IndexList{{Index(i)}});
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

    //
    // Extents and bases constructor.
    //
    {
      const IndexList bases = {{5}};
      Range range(extents, bases);
      // Beginning of the range.
      Iterator begin = Iterator::begin(range);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = begin;
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(range);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = end;
        assert(i == end);
      }

      assert(begin.range() == range);
      assert(end.range() == range);
      assert(*begin == IndexList{{bases[0]}});
      assert(*end == IndexList{{Index(bases[0] + size)}});

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
          assert(begin[i] == IndexList{{Index(bases[0] + i)}});
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

    //
    // Extents, bases, and steps constructor.
    //
    {
      const IndexList bases = {{5}};
      const IndexList steps = {{3}};
      Range range(extents, bases, steps);
      // Beginning of the range.
      Iterator begin = Iterator::begin(range);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = begin;
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(range);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = end;
        assert(i == end);
      }

      assert(begin.range() == range);
      assert(end.range() == range);
      assert(*begin == bases);
      assert(*end == IndexList{{Index(bases[0] + extents[0] * steps[0])}});

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
          assert(begin[i] == IndexList{{Index(bases[0] + i * steps[0])}});
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
    typedef container::MultiIndexRangeIterator<2> Iterator;
    typedef Iterator::Range Range;
    typedef Iterator::SizeList SizeList;
    typedef Iterator::IndexList IndexList;
    typedef Iterator::Index Index;
    typedef Iterator::size_type size_type;

    static_assert(Iterator::Dimension == 2, "Bad dimension.");

    SizeList extents = {{2, 3}};
    const size_type size = 6;

    //
    // Range with extent constructor.
    //
    {
      Range range(extents);
      // Beginning of the range.
      Iterator begin = Iterator::begin(range);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = begin;
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(range);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = end;
        assert(i == end);
      }

      assert(begin.range() == range);
      assert(end.range() == range);
      assert(*begin == (IndexList{{0, 0}}));
      assert(*end == (IndexList{{0, Index(extents[1])}}));

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
            assert(begin[n++] == (IndexList{{Index(i), Index(j)}}));
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

    //
    // Extents and bases constructor.
    //
    {
      const IndexList bases = {{5, 7}};
      Range range(extents, bases);
      // Beginning of the range.
      Iterator begin = Iterator::begin(range);
      {
        // Copy constructor.
        Iterator i(begin);
        assert(i == begin);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = begin;
        assert(i == begin);
      }

      // End of the range.
      Iterator end = Iterator::end(range);
      {
        // Copy constructor.
        Iterator i(end);
        assert(i == end);
      }
      {
        // Assignment operator.
        Iterator i = Iterator::begin(range);
        i = end;
        assert(i == end);
      }

      assert(begin.range() == range);
      assert(end.range() == range);
      assert(*begin == bases);
      assert(*end == (IndexList{{bases[0], bases[1] + Index(extents[1])}}));

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
            assert(begin[n++] == bases + (IndexList{{Index(i), Index(j)}}));
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

// -*- C++ -*-

#include "stlib/container/SimpleMultiIndexRangeIterator.h"

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  //--------------------------------------------------------------------------
  // 1-D.
  //--------------------------------------------------------------------------
  {
    typedef container::SimpleMultiIndexRangeIterator<1> Iterator;
    typedef Iterator::Range Range;
    typedef Iterator::IndexList IndexList;
    typedef Iterator::Index Index;

    static_assert(Iterator::Dimension == 1, "Bad dimension.");

    const Index size = 7;
    IndexList extents = {{size}};

    //
    // Range with extent constructor.
    //
    {
      Range range = {extents, {{0}}};

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
        std::size_t count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        for (std::size_t i = 0; i != size; ++i) {
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
        assert(end.base() == size);
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

      assert(std::distance(begin, end) == std::ptrdiff_t(size));
      assert(size + begin == end);
    }

    //
    // Extents and bases constructor.
    //
    {
      const IndexList bases = {{5}};
      Range range = {extents, bases};
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
      assert(*end == IndexList{{bases[0] + size}});

      {
        // Pre-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        for (std::size_t i = 0; i != size; ++i) {
          assert(begin[i] == IndexList{{bases[0] + i}});
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
        assert(end.base() == size);
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

      assert(std::distance(begin, end) == std::ptrdiff_t(size));
      assert(size + begin == end);
    }
  }

  //--------------------------------------------------------------------------
  // 2-D.
  //--------------------------------------------------------------------------
  {
    typedef container::SimpleMultiIndexRangeIterator<2> Iterator;
    typedef Iterator::Range Range;
    typedef Iterator::IndexList IndexList;

    static_assert(Iterator::Dimension == 2, "Bad dimension.");

    IndexList extents = {{2, 3}};
    const std::size_t size = 6;

    //
    // Range with extent constructor.
    //
    {
      Range range = {extents, {{0, 0}}};
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
      assert(*end == (IndexList{{0, extents[1]}}));

      {
        // Pre-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        std::size_t n = 0;
        for (std::size_t j = 0; j != extents[1]; ++j) {
          for (std::size_t i = 0; i != extents[0]; ++i) {
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
        assert(end.base() == size);
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

      assert(std::distance(begin, end) == std::ptrdiff_t(size));
      assert(size + begin == end);
    }

    //
    // Extents and bases constructor.
    //
    {
      const IndexList bases = {{5, 7}};
      Range range = {extents, bases};
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
      assert(*end == (IndexList{{bases[0], bases[1] + extents[1]}}));

      {
        // Pre-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; ++i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-increment.
        std::size_t count = 0;
        for (Iterator i = begin; i != end; i++) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Pre-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; --i) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Post-decrement.
        std::size_t count = 0;
        for (Iterator i = end; i != begin; i--) {
          ++count;
        }
        assert(count == size);
      }
      {
        // Iterator indexing.
        std::size_t n = 0;
        for (std::size_t j = 0; j != extents[1]; ++j) {
          for (std::size_t i = 0; i != extents[0]; ++i) {
            assert(begin[n++] == bases + (IndexList{{i, j}}));
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
        assert(end.base() == size);
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

      assert(std::distance(begin, end) == std::ptrdiff_t(size));
      assert(size + begin == end);
    }
  }

  return 0;
}

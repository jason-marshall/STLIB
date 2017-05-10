// -*- C++ -*-

#include "stlib/ads/array/ArrayIndexIterator.h"
#include "stlib/ads/array/Array.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  //-------------------------------------------------------------------------
  // 1-D
  //-------------------------------------------------------------------------

  {
    Array<1> array(Array<1>::range_type(-5, 5));
    Array<1>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      ArrayIndexIterator<1> x(array);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        const ArrayIndexIterator<1> end = array.indices_end();
        for (x = array.indices_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      ArrayIndexIterator<1> x(array);
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }

      // Begin and end accessors.
      {
        int i = array.size() - 1;
        const ArrayIndexIterator<1> begin = array.indices_begin();
        for (x = array.indices_end() - 1; x >= begin; --x, --i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }
    // Random access iterator.
    {
      // Indexing.
      ArrayIndexIterator<1> x(array);
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == x[i]);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<1> y(x);
        y += i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x + i));
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<1> y(x);
        y -= -i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x - (-i)));
      }
    }
  }


  //-------------------------------------------------------------------------
  // 2-D
  //-------------------------------------------------------------------------
  {
    Array<2> array(Array<2>::range_type(-5, -7, 11, 13));
    Array<2>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      ArrayIndexIterator<2> x(array);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        const ArrayIndexIterator<2> end = array.indices_end();
        for (x = array.indices_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      ArrayIndexIterator<2> x(array);
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }

      // Begin and end accessors.
      {
        int i = array.size() - 1;
        const ArrayIndexIterator<2> begin = array.indices_begin();
        for (x = array.indices_end() - 1; x >= begin; --x, --i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }
    // Random access iterator.
    {
      // Indexing.
      ArrayIndexIterator<2> x(array);
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == x[i]);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<2> y(x);
        y += i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x + i));
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<2> y(x);
        y -= -i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x - (-i)));
      }
    }
  }

  //-------------------------------------------------------------------------
  // 3-D
  //-------------------------------------------------------------------------
  {
    Array<3> array(Array<3>::range_type(-2, -3, -5, 7, 11, 13));
    Array<3>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      ArrayIndexIterator<3> x(array);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x = array.indices_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        const ArrayIndexIterator<3> end = array.indices_end();
        for (x = array.indices_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      ArrayIndexIterator<3> x(array);
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x = array.indices_end() - 1;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }

      // Begin and end accessors.
      {
        int i = array.size() - 1;
        const ArrayIndexIterator<3> begin = array.indices_begin();
        for (x = array.indices_end() - 1; x >= begin; --x, --i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }
    // Random access iterator.
    {
      // Indexing.
      ArrayIndexIterator<3> x(array);
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == x[i]);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<3> y(x);
        y += i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Positive offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x + i));
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        ArrayIndexIterator<3> y(x);
        y -= -i;
        array.index_to_indices(i, index);
        assert(index == *y);
      }

      // Negative offseting.
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *(x - (-i)));
      }
    }
  }

  return 0;
}

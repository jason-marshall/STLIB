// -*- C++ -*-

#include "stlib/ads/array/IndexIterator.h"
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
    IndexRange<1> index_range(-5, 5);
    Array<1> array(index_range);
    Array<1>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      IndexIterator<1> x(index_range);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }
      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        x.set_begin();
        IndexIterator<1> end(index_range);
        end.set_end();
        for (x.set_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      IndexIterator<1> x(index_range);
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }
    }
  }

  //-------------------------------------------------------------------------
  // 2-D
  //-------------------------------------------------------------------------
  {
    IndexRange<2> index_range(-5, -7, 11, 13);
    Array<2> array(index_range);
    Array<2>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      IndexIterator<2> x(index_range);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        x.set_begin();
        IndexIterator<2> end(index_range);
        end.set_end();
        for (x.set_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      IndexIterator<2> x(index_range);
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }
    }
  }

  //-------------------------------------------------------------------------
  // 3-D
  //-------------------------------------------------------------------------
  {
    IndexRange<3> index_range(-2, -3, -5, 7, 11, 13);
    Array<3> array(index_range);
    Array<3>::index_type index;

    // Forward iterator.
    {
      // Pre-increment.
      IndexIterator<3> x(index_range);
      for (int i = 0; i != array.size(); ++i, ++x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i, x++) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-increment.
      x.set_begin();
      for (int i = 0; i != array.size(); ++i) {
        array.index_to_indices(i, index);
        assert(index == *x++);
      }

      // Begin and end accessors.
      {
        int i = 0;
        x.set_begin();
        IndexIterator<3> end(index_range);
        end.set_end();
        for (x.set_begin(); x != end; ++x, ++i) {
          array.index_to_indices(i, index);
          assert(index == *x);
        }
      }
    }

    // Bi-directional iterator.
    {
      // Pre-decrement.
      IndexIterator<3> x(index_range);
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, --x) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i, x--) {
        array.index_to_indices(i, index);
        assert(index == *x);
      }

      // Post-decrement.
      x.set_end();
      --x;
      for (int i = array.size() - 1; i >= 0; --i) {
        array.index_to_indices(i, index);
        assert(index == *x--);
      }
    }
  }

  return 0;
}

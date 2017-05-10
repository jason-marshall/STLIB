// -*- C++ -*-

#include "stlib/container/MultiArray.h"

using namespace stlib;

template<typename _MultiArray>
void
testConst(_MultiArray& array)
{
  typedef container::MultiViewIterator<_MultiArray, true> ConstIterator;
  typedef typename ConstIterator::Index Index;
  typedef typename ConstIterator::size_type size_type;

  static_assert(size_type(ConstIterator::Dimension) ==
                size_type(_MultiArray::Dimension), "Bad dimension.");

  const size_type size = array.size();

  // Const iterator
  {
    // Beginning of the range.
    ConstIterator begin = ConstIterator::begin(array);
    {
      // Copy constructor.
      ConstIterator i(begin);
      assert(i == begin);
    }
    {
      // Assignment operator.
      ConstIterator i = ConstIterator::begin(array);
      i = begin;
      assert(i == begin);
    }

    // End of the range.
    ConstIterator end = ConstIterator::end(array);
    {
      // Copy constructor.
      ConstIterator i(end);
      assert(i == end);
    }
    {
      // Assignment operator.
      ConstIterator i = ConstIterator::begin(array);
      i = end;
      assert(i == end);
    }

    assert(begin.array() == &array);
    assert(end.array() == &array);
    assert(*begin == *array.begin());

    {
      // Pre-increment.
      size_type count = 0;
      for (ConstIterator i = begin; i != end; ++i) {
        ++count;
      }
      assert(count == size);
    }
    {
      // Post-increment.
      size_type count = 0;
      for (ConstIterator i = begin; i != end; i++) {
        ++count;
      }
      assert(count == size);
    }
    {
      // Pre-decrement.
      size_type count = 0;
      for (ConstIterator i = end; i != begin; --i) {
        ++count;
      }
      assert(count == size);
    }
    {
      // Post-decrement.
      size_type count = 0;
      for (ConstIterator i = end; i != begin; i--) {
        ++count;
      }
      assert(count == size);
    }
    {
      // Iterator indexing.
      for (size_type i = 0; i != size; ++i) {
        assert(begin[i] == array[i]);
      }
    }
    {
      // +=
      ConstIterator i = begin;
      i += size;
      assert(i == end);
    }
    {
      // +
      assert(begin + size == end);
    }
    {
      // -=
      ConstIterator i = end;
      i -= size;
      assert(i == begin);
    }
    {
      // -
      assert(begin == end - size);
    }
    {
      // base
      assert(begin.base() == array.begin());
      assert(end.base() == array.end());
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

template<typename _MultiArray>
void
testNonConst(_MultiArray& array)
{
  typedef container::MultiViewIterator<_MultiArray, true> ConstIterator;
  typedef typename ConstIterator::Index Index;
  typedef typename ConstIterator::size_type size_type;
  typedef container::MultiViewIterator<_MultiArray, false> Iterator;

  const size_type size = array.size();

  //
  // Iterator
  //
  {
    // Beginning of the range.
    Iterator begin = Iterator::begin(array);
    {
      // Copy constructor.
      Iterator i(begin);
      assert(i == begin);
    }
    {
      // Assignment operator.
      Iterator i = Iterator::begin(array);
      i = begin;
      assert(i == begin);
    }

    // End of the range.
    Iterator end = Iterator::end(array);
    {
      // Copy constructor.
      Iterator i(end);
      assert(i == end);
    }
    {
      // Assignment operator.
      Iterator i = Iterator::begin(array);
      i = end;
      assert(i == end);
    }

    assert(begin.array() == &array);
    assert(end.array() == &array);
    assert(*begin == *array.begin());

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
        assert(begin[i] == array[i]);
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
      assert(begin.base() == array.begin());
      assert(end.base() == array.end());
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
  // ConstIterator/Iterator
  //
  {
    Iterator a = Iterator::begin(array);
    Iterator b = Iterator::end(array);
    // Copy constructor.
    ConstIterator i = a;
    // Assignment operator.
    ConstIterator j(i);
    j = b;

    assert(a == i);
    assert(a != j);
    assert(a < j);
    assert(a <= i);
    assert(a <= j);
    assert(b > i);
    assert(b >= j);
    assert(b >= i);

    assert(j - a == Index(size));
    assert(b - i == Index(size));
  }
}

template<typename _MultiArray>
void
test(_MultiArray& array)
{
  testConst(array);
  testNonConst(array);
}

int
main()
{
  //--------------------------------------------------------------------------
  // 1-D.
  //--------------------------------------------------------------------------
  {
    typedef container::MultiArray<int, 1> MultiArray;
    typedef MultiArray::size_type size_type;
    typedef MultiArray::SizeList SizeList;

    MultiArray array(SizeList{{7}});
    for (size_type i = 0; i != array.size(); ++i) {
      array[i] = i;
    }

    test(array);
    {
      container::MultiArrayRef<int, 1> x(array);
      test(x);
    }
    {
      container::MultiArrayConstRef<int, 1> x(array);
      testConst(x);
    }
  }

  //--------------------------------------------------------------------------
  // 2-D.
  //--------------------------------------------------------------------------
  {
    typedef container::MultiArray<int, 2> MultiArray;
    typedef MultiArray::size_type size_type;
    typedef MultiArray::SizeList SizeList;

    MultiArray array(SizeList{{2, 3}});
    for (size_type i = 0; i != array.size(); ++i) {
      array[i] = i;
    }
    test(array);
  }

  return 0;
}

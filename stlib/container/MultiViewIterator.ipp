// -*- C++ -*-

#if !defined(__container_MultiViewIterator_ipp__)
#error This file is an implementation detail of the class MultiViewIterator.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Return an iterator to the beginning of the index range.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>
MultiViewIterator<_MultiArray, _IsConst>::
begin(MultiArrayReference array)
{
  MultiViewIterator x;
  x._indexList = array.bases();
  x._rank = 0;
  x._iterator = array.data();
  x._array = &array;
  return x;
}

// Return an iterator to the end of the index range.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>
MultiViewIterator<_MultiArray, _IsConst>::
end(MultiArrayReference array)
{
  MultiViewIterator x;
  x._indexList = array.bases();
  const size_type n = array.storage()[Dimension - 1];
  x._indexList[n] = array.bases()[n] + array.extents()[n];
  x._rank = array.size();
  x._iterator = array.data() + array.extents()[n] * array.strides()[n];
  x._array = &array;
  return x;
}

// Copy constructor from non-const.
template<typename _MultiArray, bool _IsConst>
template<bool _IsConst2>
inline
MultiViewIterator<_MultiArray, _IsConst>::
MultiViewIterator(const MultiViewIterator<MultiArray, _IsConst2>& other) :
  _indexList(other.indexList()),
  _rank(other.rank()),
  _iterator(other.base()),
  _array(other.array())
{
}

// Assignment operator from non-const.

template<typename _MultiArray, bool _IsConst>
template<bool _IsConst2>
inline
MultiViewIterator<_MultiArray, _IsConst>&
MultiViewIterator<_MultiArray, _IsConst>::
operator=(const MultiViewIterator<MultiArray, _IsConst2>& other)
{
  _indexList = other.indexList();
  _rank = other.rank();
  _iterator = other.base();
  _array = other.array();
  return *this;
}

//--------------------------------------------------------------------------
// Validity.

// Return true if the iterator is valid.
template<typename _MultiArray, bool _IsConst>
inline
bool
MultiViewIterator<_MultiArray, _IsConst>::
isValid() const
{
  if (!(0 <= _rank && _rank < Index(_array->size()))) {
    return false;
  }
  for (size_type i = 0; i != Dimension; ++i) {
    if (!(_array->bases()[i] <= _indexList[i] &&
          _indexList[i] < _array->bases()[i] +
          Index(_array->extents()[i]))) {
      return false;
    }
  }
  return true;
}

// Return true if the iterator is at the beginning.
template<typename _MultiArray, bool _IsConst>
inline
bool
MultiViewIterator<_MultiArray, _IsConst>::
isBegin() const
{
  return _indexList == _array->bases();
}

// Return true if the iterator is at the end.
template<typename _MultiArray, bool _IsConst>
inline
bool
MultiViewIterator<_MultiArray, _IsConst>::
isEnd() const
{
  for (size_type i = 0; i != Dimension - 1; ++i) {
    const size_type n = _array->storage()[i];
    if (_indexList[n] != _array->bases()[n]) {
      return false;
    }
  }
  const size_type n = _array->storage()[Dimension - 1];
  if (_indexList[n] != _array->bases()[n] +
      Index(_array->extents()[n])) {
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------
// Forward iterator requirements.

// Pre-increment.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>&
MultiViewIterator<_MultiArray, _IsConst>::
operator++()
{
#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  // Increment the index list.
  const size_type s0 = _array->storage()[0];
  ++_indexList[s0];
  _iterator += _array->strides()[s0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    const size_type s = _array->storage()[i];
    if (_indexList[s] == _array->bases()[s] + Index(_array->extents()[s])) {
      const size_type s1 = _array->storage()[i + 1];
      _indexList[s] = _array->bases()[s];
      ++_indexList[s1];
      _iterator += _array->strides()[s1] -
                   _array->strides()[s] * _array->extents()[s];
    }
    else {
      break;
    }
  }
  // Increment the rank.
  ++_rank;

#ifdef STLIB_DEBUG
  assert(isValid() || isEnd());
#endif

  return *this;
}

// Post-increment.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>
MultiViewIterator<_MultiArray, _IsConst>::
operator++(int)
{
  MultiViewIterator tmp(*this);
  ++*this;
  return tmp;
}

//--------------------------------------------------------------------------
// Bidirectional iterator requirements.

// Pre-decrement.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>&
MultiViewIterator<_MultiArray, _IsConst>::
operator--()
{
#ifdef STLIB_DEBUG
  assert((isValid() || isEnd()) && ! isBegin());
#endif

  // Decrement the index list.
  const size_type s0 = _array->storage()[0];
  --_indexList[s0];
  _iterator -= _array->strides()[s0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    const size_type s = _array->storage()[i];
    if (_indexList[s] == _array->bases()[s] - 1) {
      const size_type s1 = _array->storage()[i + 1];
      _indexList[s] = _array->bases()[s] + _array->extents()[s] - 1;
      --_indexList[s1];
      _iterator -= _array->strides()[s1] -
                   _array->strides()[s] * _array->extents()[s];
    }
    else {
      break;
    }
  }
  // Decrement the rank.
  --_rank;

#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  return *this;
}


// Post-decrement.
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>
MultiViewIterator<_MultiArray, _IsConst>::
operator--(int)
{
  MultiViewIterator tmp(*this);
  --*this;
  return tmp;
}

// Calculate the index list from the rank.
template<typename _MultiArray, bool _IsConst>
inline
void
MultiViewIterator<_MultiArray, _IsConst>::
update()
{
#ifdef STLIB_DEBUG
  assert(0 <= _rank && _rank <= Index(_array->size()));
#endif
  Index r = _rank;
  for (std::size_t i = 0; i != Dimension; ++i) {
    // Traverse from most significant to least.
    const std::size_t n = _array->storage()[Dimension - i - 1];
    _indexList[n] = r / _array->strides()[n];
    r -= _indexList[n] * _array->strides()[n];
    _indexList[n] += _array->bases()[n];
  }
  _iterator = _array->data() + _array->arrayIndex(_indexList);
#ifdef STLIB_DEBUG
  assert(r == 0);
#endif
}

} // namespace container
}

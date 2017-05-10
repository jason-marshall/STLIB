// -*- C++ -*-

#if !defined(__container_SimpleMultiIndexRangeIterator_ipp__)
#error This file is an implementation detail of the class SimpleMultiIndexRangeIterator.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Return an iterator to the beginning of the index range.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
begin(const Range& range)
{
  SimpleMultiIndexRangeIterator x;
  x._indexList = range.bases;
  x._rank = 0;
  x._range = range;
  return x;
}

// Return an iterator to the beginning of the index range.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
begin(const IndexList& extents)
{
  Range range = {extents, ext::filled_array<IndexList>(0)};
  return begin(range);
}

// Return an iterator to the end of the index range.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
end(const Range& range)
{
  SimpleMultiIndexRangeIterator x;
  x._indexList = range.bases;
  // Column-major
  x._indexList[Dimension - 1] = range.bases[Dimension - 1] +
                                range.extents[Dimension - 1];
#if 0
  // Row-major
  x._indexList[0] = range.bases[0] + range.extents[0];
#endif
  x._rank = ext::product(range.extents);
  x._range = range;
  return x;
}

// Return an iterator to the end of the index range.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
end(const IndexList& extents)
{
  Range range = {extents, ext::filled_array<IndexList>(0)};
  return end(range);
}

//--------------------------------------------------------------------------
// Validity.

// Return true if the iterator is valid.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexRangeIterator<_Dimension>::
isValid() const
{
  if (_rank >= ext::product(_range.extents)) {
    return false;
  }
  for (std::size_t i = 0; i != Dimension; ++i) {
    if (!(_range.bases[i] <= _indexList[i] &&
          _indexList[i] < _range.bases[i] + _range.extents[i])) {
      return false;
    }
  }
  return true;
}

// Return true if the iterator is at the beginning.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexRangeIterator<_Dimension>::
isBegin() const
{
  if (_rank == 0) {
    assert(_indexList == _range.bases);
    return true;
  }
  return false;
}

// Return true if the iterator is at the end.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexRangeIterator<_Dimension>::
isEnd() const
{
  // Column-major
  if (_indexList[Dimension - 1] != _range.bases[Dimension - 1] +
      _range.extents[Dimension - 1]) {
    return false;
  }
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] != _range.bases[i]) {
      return false;
    }
  }
#if 0
  // Row-major
  if (_indexList[0] != _range.bases[0] + _range.extents[0]) {
    return false;
  }
  for (std::size_t i = 1; i != Dimension; ++i) {
    if (_indexList[i] != _range.bases[i]) {
      return false;
    }
  }
#endif
  assert(_rank == ext::product(_range.extents));
  return true;
}

//--------------------------------------------------------------------------
// Forward iterator requirements.

// Pre-increment.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>&
SimpleMultiIndexRangeIterator<_Dimension>::
operator++()
{
#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  // Increment the index list using column-major ordering.
  ++_indexList[0];
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] == _range.bases[i] + _range.extents[i]) {
      _indexList[i] = _range.bases[i];
      ++_indexList[i + 1];
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
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
operator++(int)
{
  SimpleMultiIndexRangeIterator tmp(*this);
  ++*this;
  return tmp;
}

//--------------------------------------------------------------------------
// Bidirectional iterator requirements.

// Pre-decrement.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>&
SimpleMultiIndexRangeIterator<_Dimension>::
operator--()
{
#ifdef STLIB_DEBUG
  assert((isValid() || isEnd()) && ! isBegin());
#endif

  // Decrement the index list using column-major ordering.
  --_indexList[0];
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] + 1 == _range.bases[i]) {
      _indexList[i] = _range.bases[i] + _range.extents[i] - 1;
      --_indexList[i + 1];
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
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
SimpleMultiIndexRangeIterator<_Dimension>::
operator--(int)
{
  SimpleMultiIndexRangeIterator tmp(*this);
  --*this;
  return tmp;
}

// Calculate the index list from the rank.
template<std::size_t _Dimension>
inline
void
SimpleMultiIndexRangeIterator<_Dimension>::
calculateIndexList()
{
#ifdef STLIB_DEBUG
  assert(_rank <= ext::product(_range.extents));
#endif
  // Column-major.
  // The strides.
  IndexList strides;
  strides[0] = 1;
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    strides[i + 1] = strides[i] * _range.extents[i];
  }
  Index r = _rank;
  // Traverse from most significant to least.
  for (std::size_t i = Dimension; i != 0;) {
    --i;
    _indexList[i] = r / strides[i];
    r -= _indexList[i] * strides[i];
    _indexList[i] += _range.bases[i];
  }
#ifdef STLIB_DEBUG
  assert(r == 0);
#endif
}

} // namespace container
}

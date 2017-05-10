// -*- C++ -*-

#if !defined(__container_SimpleMultiIndexExtentsIterator_ipp__)
#error This file is an implementation detail of the class SimpleMultiIndexExtentsIterator.
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
SimpleMultiIndexExtentsIterator<_Dimension>
SimpleMultiIndexExtentsIterator<_Dimension>::
begin(const IndexList& extents)
{
  SimpleMultiIndexExtentsIterator x;
  std::fill(x._indexList.begin(), x._indexList.end(), 0);
  x._rank = 0;
  x._extents = extents;
  return x;
}

// Return an iterator to the end of the index range.
template<std::size_t _Dimension>
inline
SimpleMultiIndexExtentsIterator<_Dimension>
SimpleMultiIndexExtentsIterator<_Dimension>::
end(const IndexList& extents)
{
  SimpleMultiIndexExtentsIterator x;
  std::fill(x._indexList.begin(), x._indexList.end(), 0);
  // Column-major
  x._indexList[Dimension - 1] = extents[Dimension - 1];
  x._rank = ext::product(extents);
  x._extents = extents;
  return x;
}

//--------------------------------------------------------------------------
// Validity.

// Return true if the iterator is valid.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexExtentsIterator<_Dimension>::
isValid() const
{
  if (_rank >= ext::product(_extents)) {
    return false;
  }
  for (size_type i = 0; i != Dimension; ++i) {
    if (_indexList[i] >= _extents[i]) {
      return false;
    }
  }
  return true;
}

// Return true if the iterator is at the beginning.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexExtentsIterator<_Dimension>::
isBegin() const
{
  return _rank == 0;
}

// Return true if the iterator is at the end.
template<std::size_t _Dimension>
inline
bool
SimpleMultiIndexExtentsIterator<_Dimension>::
isEnd() const
{
  // Column-major
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] != 0) {
      return false;
    }
  }
  if (_indexList[Dimension - 1] != _extents[Dimension - 1]) {
    return false;
  }
#ifdef STLIB_DEBUG
  assert(_rank == ext::product(_extents));
#endif
  return true;
}

//--------------------------------------------------------------------------
// Forward iterator requirements.

// Pre-increment.
template<std::size_t _Dimension>
inline
SimpleMultiIndexExtentsIterator<_Dimension>&
SimpleMultiIndexExtentsIterator<_Dimension>::
operator++()
{
#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  // Increment the index list using column-major ordering.
  ++_indexList[0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] == _extents[i]) {
      _indexList[i] = 0;
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
SimpleMultiIndexExtentsIterator<_Dimension>
SimpleMultiIndexExtentsIterator<_Dimension>::
operator++(int)
{
  SimpleMultiIndexExtentsIterator tmp(*this);
  ++*this;
  return tmp;
}

//--------------------------------------------------------------------------
// Bidirectional iterator requirements.

// Pre-decrement.
template<std::size_t _Dimension>
inline
SimpleMultiIndexExtentsIterator<_Dimension>&
SimpleMultiIndexExtentsIterator<_Dimension>::
operator--()
{
#ifdef STLIB_DEBUG
  assert((isValid() || isEnd()) && ! isBegin());
#endif

  // Decrement the index list using column-major ordering.
  --_indexList[0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] == Index(-1)) {
      _indexList[i] = _extents[i] - 1;
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
SimpleMultiIndexExtentsIterator<_Dimension>
SimpleMultiIndexExtentsIterator<_Dimension>::
operator--(int)
{
  SimpleMultiIndexExtentsIterator tmp(*this);
  --*this;
  return tmp;
}

// Calculate the index list from the rank.
template<std::size_t _Dimension>
inline
void
SimpleMultiIndexExtentsIterator<_Dimension>::
calculateIndexList()
{
#ifdef STLIB_DEBUG
  assert(_rank <= Index(ext::product(_extents)));
#endif
  // Column-major.
  // The strides.
  IndexList strides;
  strides[0] = 1;
  for (size_type i = 0; i != Dimension - 1; ++i) {
    strides[i + 1] = strides[i] * _extents[i];
  }
  Index r = _rank;
  // Traverse from most significant to least.
  for (std::size_t i = Dimension; i != 0;) {
    --i;
    _indexList[i] = r / strides[i];
    r -= _indexList[i] * strides[i];
  }
#ifdef STLIB_DEBUG
  assert(r == 0);
#endif
}

} // namespace container
}

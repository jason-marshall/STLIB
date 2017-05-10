// -*- C++ -*-

#if !defined(__container_MultiIndexRangeIterator_ipp__)
#error This file is an implementation detail of the class MultiIndexRangeIterator.
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
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
begin(const Range& range)
{
  MultiIndexRangeIterator x;
  x._indexList = range.bases();
  x._rank = 0;
  x._range = range;
  return x;
}

// Return an iterator to the beginning of the index range.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
begin(const SizeList& extents)
{
  return begin(Range(extents));
}

// Return an iterator to the end of the index range.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
end(const Range& range)
{
  MultiIndexRangeIterator x;
  x._indexList = range.bases();
  // Column-major
  x._indexList[Dimension - 1] = range.bases()[Dimension - 1] +
                                range.extents()[Dimension - 1] * range.steps()[Dimension - 1];
#if 0
  // Row-major
  x._indexList[0] = range.bases()[0] + range.extents()[0] * range.steps()[0];
#endif
  x._rank = ext::product(range.extents());
  x._range = range;
  return x;
}

// Return an iterator to the end of the index range.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
end(const SizeList& extents)
{
  return end(Range(extents));
}

// Copy constructor.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>::
MultiIndexRangeIterator(const MultiIndexRangeIterator& other) :
  _indexList(other._indexList),
  _rank(other._rank),
  _range(other._range)
{
}

// Assignment operator.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>&
MultiIndexRangeIterator<_Dimension>::
operator=(const MultiIndexRangeIterator& other)
{
  if (this != &other) {
    _indexList = other._indexList;
    _rank = other._rank;
    _range = other._range;
  }
  return *this;
}

//--------------------------------------------------------------------------
// Validity.

// Return true if the iterator is valid.
template<std::size_t _Dimension>
inline
bool
MultiIndexRangeIterator<_Dimension>::
isValid() const
{
  if (!(0 <= _rank && _rank < Index(ext::product(_range.extents())))) {
    return false;
  }
  for (size_type i = 0; i != Dimension; ++i) {
    if (!(_range.bases()[i] <= _indexList[i] &&
          _indexList[i] < _range.bases()[i] +
          Index(_range.extents()[i]) * _range.steps()[i])) {
      return false;
    }
  }
  return true;
}

// Return true if the iterator is at the beginning.
template<std::size_t _Dimension>
inline
bool
MultiIndexRangeIterator<_Dimension>::
isBegin() const
{
  if (_rank == 0) {
    assert(_indexList == _range.bases());
    return true;
  }
  return false;
}

// Return true if the iterator is at the end.
template<std::size_t _Dimension>
inline
bool
MultiIndexRangeIterator<_Dimension>::
isEnd() const
{
  // Column-major
  if (_indexList[Dimension - 1] != _range.bases()[Dimension - 1] +
      Index(_range.extents()[Dimension - 1]) * _range.steps()[Dimension - 1]) {
    return false;
  }
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] != _range.bases()[i]) {
      return false;
    }
  }
#if 0
  // Row-major
  if (_indexList[0] != _range.bases()[0] +
      Index(_range.extents()[0]) * _range.steps()[0]) {
    return false;
  }
  for (size_type i = 1; i != Dimension; ++i) {
    if (_indexList[i] != _range.bases()[i]) {
      return false;
    }
  }
#endif
  assert(_rank == Index(ext::product(_range.extents())));
  return true;
}

//--------------------------------------------------------------------------
// Forward iterator requirements.

// Pre-increment.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>&
MultiIndexRangeIterator<_Dimension>::
operator++()
{
#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  // Increment the index list using column-major ordering.
  _indexList[0] += _range.steps()[0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] == _range.bases()[i] +
        Index(_range.extents()[i]) * _range.steps()[i]) {
      _indexList[i] = _range.bases()[i];
      _indexList[i + 1] += _range.steps()[i + 1];
    }
    else {
      break;
    }
  }
#if 0
  // Increment the index list using row-major ordering.
  _indexList[Dimension - 1] += _range.steps()[Dimension - 1];
  for (size_type i = Dimension - 1; i != 0; --i) {
    if (_indexList[i] == _range.bases()[i] +
        Index(_range.extents()[i]) * _range.steps()[i]) {
      _indexList[i] = _range.bases()[i];
      _indexList[i - 1] += _range.steps()[i - 1];
    }
    else {
      break;
    }
  }
#endif
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
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
operator++(int)
{
  MultiIndexRangeIterator tmp(*this);
  ++*this;
  return tmp;
}

//--------------------------------------------------------------------------
// Bidirectional iterator requirements.

// Pre-decrement.
template<std::size_t _Dimension>
inline
MultiIndexRangeIterator<_Dimension>&
MultiIndexRangeIterator<_Dimension>::
operator--()
{
#ifdef STLIB_DEBUG
  assert((isValid() || isEnd()) && ! isBegin());
#endif

  // Decrement the index list using column-major ordering.
  _indexList[0] -= _range.steps()[0];
  for (size_type i = 0; i != Dimension - 1; ++i) {
    if (_indexList[i] == _range.bases()[i] - _range.steps()[i]) {
      _indexList[i] = _range.bases()[i] +
                      (_range.extents()[i] - 1) * _range.steps()[i];
      _indexList[i + 1] -= _range.steps()[i + 1];
    }
    else {
      break;
    }
  }
#if 0
  // Decrement the index list using row-major ordering.
  _indexList[Dimension - 1] -= _range.steps()[Dimension - 1];
  for (size_type i = Dimension - 1; i != 0; --i) {
    if (_indexList[i] == _range.bases()[i] - _range.steps()[i]) {
      _indexList[i] = _range.bases()[i] +
                      (_range.extents()[i] - 1) * _range.steps()[i];
      _indexList[i - 1] -= _range.steps()[i - 1];
    }
    else {
      break;
    }
  }
#endif
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
MultiIndexRangeIterator<_Dimension>
MultiIndexRangeIterator<_Dimension>::
operator--(int)
{
  MultiIndexRangeIterator tmp(*this);
  --*this;
  return tmp;
}

// Calculate the index list from the rank.
template<std::size_t _Dimension>
inline
void
MultiIndexRangeIterator<_Dimension>::
calculateIndexList()
{
#ifdef STLIB_DEBUG
  assert(0 <= _rank && _rank <= Index(ext::product(_range.extents())));
#endif
  // Column-major.
  // The strides.
  IndexList strides;
  strides[0] = 1;
  for (size_type i = 0; i != Dimension - 1; ++i) {
    strides[i + 1] = strides[i] * _range.extents()[i];
  }
  Index r = _rank;
  // Traverse from most significant to least.
  for (std::size_t i = Dimension; i != 0;) {
    --i;
    _indexList[i] = r / strides[i];
    r -= _indexList[i] * strides[i];
    _indexList[i] *= _range.steps()[i];
    _indexList[i] += _range.bases()[i];
  }
#if 0
  // Row-major.
  // The strides.
  IndexList strides;
  strides[Dimension - 1] = 1;
  for (size_type i = Dimension - 1; i != 0; --i) {
    strides[i - 1] = strides[i] * _range.extents()[i];
  }
  Index r = _rank;
  // Traverse from most significant to least.
  for (std::size_t i = 0; i != Dimension; ++i) {
    _indexList[i] = r / strides[i];
    r -= _indexList[i] * strides[i];
    _indexList[i] *= _range.steps()[i];
    _indexList[i] += _range.bases()[i];
  }
#endif
#ifdef STLIB_DEBUG
  assert(r == 0);
#endif
}

} // namespace container
}

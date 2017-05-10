// -*- C++ -*-

#if !defined(__container_MultiIndexRange_ipp__)
#error This file is an implementation detail of the class MultiIndexRange.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

#if 0
// Construct from the lower and upper bounds, and optionally the steps.
template<std::size_t _Dimension>
inline
MultiIndexRange<_Dimension>::
MultiIndexRange(const IndexList& lower, const IndexList& upper,
                const IndexList& steps) :
  _extents(),
  _bases(lower),
  _steps(steps)
{
  _extents = upper - lower;
  for (size_type i = 0; i != Dimension; ++i) {
    _extents[i] = _extents[i] / _steps[i] +
                  (_extents[i] % _steps[i] ? 1 : 0);
  }
#ifdef STLIB_DEBUG
  for (size_type i = 0; i != Dimension; ++i) {
    assert(_extents[i] >= 0);
    assert(steps[i] > 0);
  }
#endif
}
#endif

//---------------------------------------------------------------------------
// Free functions.

// Return the intersection of the two ranges.
template<std::size_t _Dimension>
inline
MultiIndexRange<_Dimension>
overlap(const MultiIndexRange<_Dimension>& x,
        const MultiIndexRange<_Dimension>& y)
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != _Dimension; ++i) {
    assert(x.steps()[i] == 1 && y.steps()[i] == 1);
  }
#endif
  return overlap(x.extents(), x.bases(), y.extents(), y.bases());
}

// Return the intersection of the two index ranges.
template<std::size_t _Dimension, typename _Size, typename _Index>
inline
MultiIndexRange<_Dimension>
overlap(const std::array<_Size, _Dimension>& extents1,
        const std::array<_Index, _Dimension>& bases1,
        const std::array<_Size, _Dimension>& extents2,
        const std::array<_Index, _Dimension>& bases2)
{
  typedef typename MultiIndexTypes<_Dimension>::SizeList SizeList;
  typedef typename MultiIndexTypes<_Dimension>::IndexList IndexList;
  typedef typename MultiIndexTypes<_Dimension>::Index Index;

  Index upper;
  IndexList bases;
  SizeList extents;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    bases[i] = std::max(bases1[i], bases2[i]);
    upper = std::min(bases1[i] + Index(extents1[i]),
                     bases2[i] + Index(extents2[i]));
    if (upper > bases[i]) {
      extents[i] = upper - bases[i];
    }
    else {
      extents[i] = 0;
    }
  }
  return MultiIndexRange<_Dimension>(extents, bases);
}

// Return true if the index is in the index range.
template<std::size_t _Dimension>
inline
bool
isIn(const MultiIndexRange<_Dimension>& range,
     const typename MultiIndexRange<_Dimension>::IndexList& index)
{
  typedef typename MultiIndexRange<_Dimension>::Index Index;
  // If the range has unit steps.
  if (std::count(range.steps().begin(), range.steps().end(), 1) == _Dimension) {
    for (std::size_t d = 0; d != _Dimension; ++d) {
      if (!(Index(range.bases()[d]) <= index[d] &&
            index[d] < range.bases()[d] + Index(range.extents()[d]))) {
        return false;
      }
    }
  }
  // If the range does not have unit steps.
  else {
    for (std::size_t d = 0; d != _Dimension; ++d) {
      if (!(Index(range.bases()[d]) <= index[d] &&
            index[d] < range.bases()[d] +
            range.steps()[d] * Index(range.extents()[d]) &&
            (index[d] - Index(range.bases()[d])) % range.steps()[d] == 0)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace container
}

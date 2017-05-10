// -*- C++ -*-

#if !defined(__container_SimpleMultiIndexRange_ipp__)
#error This file is an implementation detail of the class SimpleMultiIndexRange.
#endif

namespace stlib
{
namespace container
{

//---------------------------------------------------------------------------
// Free functions.

// Return the intersection of the two ranges.
template<std::size_t _Dimension>
inline
SimpleMultiIndexRange<_Dimension>
overlap(const SimpleMultiIndexRange<_Dimension>& x,
        const SimpleMultiIndexRange<_Dimension>& y)
{
  typedef typename SimpleMultiIndexRange<_Dimension>::Index Index;

  Index upper;
  SimpleMultiIndexRange<_Dimension> z;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    z.bases[i] = std::max(x.bases[i], y.bases[i]);
    upper = std::min(x.bases[i] + x.extents[i], y.bases[i] + y.extents[i]);
    if (upper > z.bases[i]) {
      z.extents[i] = upper - z.bases[i];
    }
    else {
      z.extents[i] = 0;
    }
  }
  return z;
}

// Return true if the index is in the index range.
template<std::size_t _Dimension>
inline
bool
isIn(const SimpleMultiIndexRange<_Dimension>& range,
     const typename SimpleMultiIndexRange<_Dimension>::IndexList& index)
{
  for (std::size_t d = 0; d != _Dimension; ++d) {
    if (!(range.bases[d] <= index[d] &&
          index[d] < range.bases[d] + range.extents[d])) {
      return false;
    }
  }
  return true;
}

} // namespace container
}

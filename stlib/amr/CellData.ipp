// -*- C++ -*-

#if !defined(__amr_CellData_ipp__)
#error This file is an implementation detail of the class CellData.
#endif

namespace stlib
{
namespace amr
{

//
// Constructors etc.
//

// Assignment operator.
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
CellData<_Traits, _Depth, _GhostWidth, FloatT>&
CellData<_Traits, _Depth, _GhostWidth, FloatT>::
operator=(const CellData& other)
{
  if (this != &other) {
    _array.rebuild(other._array);
  }
  return *this;
}

//
// Prolongation, restriction, and synchronization.
//

// Copy from the interior array data. The patches must be at the same level
// and adjacent.
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
void
CellData<_Traits, _Depth, _GhostWidth, FloatT>::
copy(const CellData& source)
{
  typedef container::MultiIndexRange<_Traits::Dimension> Range;

  const Range range =
    container::overlap(source.getArray().extents() - 2 * GhostWidth,
                       source.getArray().bases() + Index(GhostWidth),
                       _array.extents(), _array.bases());
  _array.view(range) = source.getArray().view(range);
}

// Prolongation from data that is one level lower.
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
void
CellData<_Traits, _Depth, _GhostWidth, FloatT>::
prolongConstant(const CellData& source)
{
  ArrayView a = getInteriorArray();
#ifdef STLIB_DEBUG
  for (std::size_t n = 0; n != _Traits::Dimension; ++n) {
    assert(a.bases()[n] / 2 >=
           source.getArray().bases()[n] + Index(CellData::GhostWidth) &&
           (a.bases()[n] + Index(a.extents()[n]) - 1) / 2
           <= source.getArray().bases()[n] +
           Index(source.getArray().extents()[n]) - 1 -
           Index(CellData::GhostWidth));
  }
#endif
  const typename ArrayView::iterator end = a.end();
  for (typename ArrayView::iterator i = a.begin(); i != end; ++i) {
    *i = source.getArray()(i.indexList() / Index(2));
  }
}

// Restriction from interior array data that is one level higher.
template < class _Traits, std::size_t _Depth, std::size_t _GhostWidth,
           typename FloatT >
inline
void
CellData<_Traits, _Depth, _GhostWidth, FloatT>::
restrictLinear(const CellData& source)
{
  // REMOVE
  //typedef container::MultiIndexRange<_Traits::Dimension> Range;

  // The interior array for the higher level patch.
  ArrayConstView a = source.getInteriorArray();
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != _Traits::Dimension; ++i) {
    // The extents must be multiples of 2.
    assert(a.extents()[i] % 2 == 0);
    assert(a.bases()[i] / 2 >= _array.bases()[i] + Index(GhostWidth));
    assert((a.bases()[i] + a.extents()[i] - 1) / 2 <=
           _array.bases()[i] + _array.extents()[i] - 1 - GhostWidth);
  }
#endif
  // The overlapping domain in the target patch.
  Range domain(a.extents() / std::size_t(2), a.bases() / Index(2));
  // The overlapping array.
  ArrayView overlap = _array.view(domain);
  // Restriction.
  overlap.fill(ext::filled_array<FieldTuple>(0));
  {
    const typename ArrayConstView::const_iterator end = a.end();
    for (typename ArrayConstView::const_iterator i = a.begin(); i != end; ++i) {
      _array(i.indexList() / Index(2)) += *i;
    }
  }
  overlap /= ext::filled_array<FieldTuple>(_Traits::NumberOfOrthants);
}

} // namespace amr
}

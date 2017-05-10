// -*- C++ -*-

#if !defined(__geom_spatialIndexing_SpatialIndex_ipp__)
#error This file is an implementation detail of the class SpatialIndex.
#endif

namespace stlib
{
namespace geom
{

//--------------------------------------------------------------------------
// Manipulators.

// Transform to the specified neighbor.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndex<_Dimension, _MaximumLevel>::
transformToNeighbor(const std::size_t n)
{
#ifdef STLIB_DEBUG
  assert(n < 2 * Dimension);
  assert(hasNeighbor(*this, n));
#endif
  // The coordinate is n / 2.
  // The direction in that coordinate is n % 2.
  // Change coordinate by +-1.
  _coordinates[n / 2] += 2 * (n % 2) - 1;
  updateCode();
}

} // namespace geom
}

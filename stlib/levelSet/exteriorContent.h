// -*- C++ -*-

#if !defined(__levelSet_exteriorContent_h__)
#define __levelSet_exteriorContent_h__

#include "stlib/levelSet/GridGeometry.h"
#include "stlib/numerical/constants.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetExteriorVolume Compute the volume of the exterior of a region.
*/
//@{


//! Compute the volume of the exterior of the union of the balls.
template<typename _T, std::size_t _D, std::size_t N>
_T
exteriorContent
(const GridGeometry<_D, N, _T>& grid,
 const std::vector<bool>& isActive,
 const std::vector<geom::Ball<_T, _D> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies);


//@}

} // namespace levelSet
}

#define __levelSet_exteriorContent_ipp__
#include "stlib/levelSet/exteriorContent.ipp"
#undef __levelSet_exteriorContent_ipp__

#endif

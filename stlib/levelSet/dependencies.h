// -*- C++ -*-

#if !defined(__levelSet_dependencies_h__)
#define __levelSet_dependencies_h__

#include "stlib/levelSet/GridGeometry.h"

#include "stlib/container/SimpleMultiIndexExtentsIterator.h"
#include "stlib/geom/kernel/Ball.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetDependencies Patch-Ball Dependencies
*/
//@{


//! Order the patch dependencies.
/*! For each of the patches, put the ball that is closest to the center first
  in the list of dependencies. This can improve performance for distance
  computations when one can abort if the distance is below a certain
  threshold. */
template<typename _T, std::size_t _D, std::size_t N>
void
putClosestBallsFirst(const GridGeometry<_D, N, _T>& grid,
                     const std::vector<geom::Ball<_T, _D> >& balls,
                     container::StaticArrayOfArrays<unsigned>* dependencies);


//@}

} // namespace levelSet
}

#define __levelSet_dependencies_ipp__
#include "stlib/levelSet/dependencies.ipp"
#undef __levelSet_dependencies_ipp__

#endif

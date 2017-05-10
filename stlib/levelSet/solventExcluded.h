// -*- C++ -*-

#if !defined(__levelSet_solventExcluded_h__)
#define __levelSet_solventExcluded_h__

#include "stlib/levelSet/boolean.h"
#include "stlib/levelSet/countGrid.h"
#include "stlib/levelSet/dependencies.h"
#include "stlib/levelSet/negativeDistance.h"
#include "stlib/levelSet/outside.h"
#include "stlib/levelSet/positiveDistance.h"
#include "stlib/levelSet/powerDistance.h"

#include "stlib/container/SimpleMultiIndexExtentsIterator.h"
#include "stlib/hj/hj.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetSolventExcluded Solvent-Excluded Surface
These functions calculate level sets related to the solvent-excluded surface.
*/
//@{


//! Determine the seeds for computing the solvent-excluded domain.
/*! We will work one patch at a time to avoid having to store the level set
  function for the solvent-accessible domain. */
template<typename _T, std::size_t _D, std::size_t N>
void
solventExcludedSeeds(const GridGeometry<_D, N, _T>& grid,
                     const std::vector<geom::Ball<_T, _D> >& balls,
                     _T probeRadius,
                     std::vector<geom::Ball<_T, _D> >* seeds);


//! Construct a level set for the solvent-excluded surface.
template<typename _T, std::size_t _D>
void
solventExcluded(GridUniform<_T, _D>* grid,
                const std::vector<geom::Ball<_T, _D> >& balls,
                _T probeRadius);


//! Construct a level set for the solvent-excluded surface.
template<typename _T, std::size_t _D, std::size_t _PatchExtent>
void
solventExcluded(Grid<_T, _D, _PatchExtent>* grid,
                const std::vector<geom::Ball<_T, _D> >& balls,
                _T probeRadius);


//@}

} // namespace levelSet
}

#define __levelSet_solventExcluded_ipp__
#include "stlib/levelSet/solventExcluded.ipp"
#undef __levelSet_solventExcluded_ipp__

#endif

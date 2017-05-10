// -*- C++ -*-

#if !defined(__levelSet_solventExcludedCavities_h__)
#define __levelSet_solventExcludedCavities_h__

#include "stlib/levelSet/solventExcluded.h"
#include "stlib/levelSet/marchingSimplices.h"
#include "stlib/levelSet/subtract.h"
#include "stlib/levelSet/exteriorContent.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetSolventExcludedCavities Solvent-Excluded Cavities
These functions calculate level sets related to the solvent-excluded cavities.
*/
//@{


//! Compute the volume and surface area for the solvent-excluded cavities.
/*! Construct an (AMR) Grid to store the level set. Use seeds to construct
  the solvent-excluded domain. Then subtract the balls. */
template<typename _T, std::size_t _D>
std::pair<_T, _T>
solventExcludedCavities(const std::vector<geom::Ball<_T, _D> >& balls,
                        _T probeRadius, _T targetGridSpacing);


//! Construct a level set for the solvent-excluded cavities.
template<typename _T, std::size_t _D>
void
solventExcludedCavities(GridUniform<_T, _D>* grid,
                        const std::vector<geom::Ball<_T, _D> >& balls,
                        _T probeRadius, _T maxDistance);


//! Construct a level set for the solvent-excluded cavities.
template<typename _T, std::size_t _D, std::size_t N>
void
solventExcludedCavities(Grid<_T, _D, N>* grid,
                        const std::vector<geom::Ball<_T, _D> >& balls,
                        _T probeRadius, _T maxDistance);


//! Construct a level set for the solvent-excluded cavities.
/*! Use seeds to construct the solvent-excluded domain. Then subtract the
 balls. */
template<typename _T, std::size_t _D, std::size_t N>
void
solventExcludedCavitiesUsingSeeds(Grid<_T, _D, N>* grid,
                                  const std::vector<geom::Ball<_T, _D> >& balls,
                                  _T probeRadius);


// Compute the content of the solvent-excluded cavities.
template<typename _T, std::size_t _D>
_T
solventExcludedCavitiesContent
(_T targetGridSpacing,
 const std::vector<geom::Ball<_T, _D> >& moleculeBalls, _T probeRadius);


//@}

} // namespace levelSet
}

#define __levelSet_solventExcludedCavities_ipp__
#include "stlib/levelSet/solventExcludedCavities.ipp"
#undef __levelSet_solventExcludedCavities_ipp__

#endif

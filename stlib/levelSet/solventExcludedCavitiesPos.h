/* -*- C++ -*- */

#ifndef __levelSet_solventExcludedCavitiesPos_h__
#define __levelSet_solventExcludedCavitiesPos_h__

// For PatchExtent.
#include "stlib/levelSet/cuda.h"
#include "stlib/levelSet/dependencies.h"
#include "stlib/levelSet/PatchActive.h"
#include "stlib/levelSet/PatchDistanceIdentifier.h"

#include "stlib/ads/algorithm/sort.h"
#include "stlib/container/vector.h"
#include "stlib/numerical/constants.h"
#include "stlib/simd/functions.h"

namespace stlib
{
namespace levelSet
{


//! Compute the volume of the solvent-excluded cavities.
/*! Avoid storing any level-set function on a grid. Only a patch at a
 time will be used.  Solvent probes will be placed by distributing
 points on a sphere (hence the Pos in the name).  Compute the volume
 using only the sign of the distance. */
float
solventExcludedCavitiesPos(const std::vector<geom::Ball<float, 3> >& balls,
                           float probeRadius, float targetGridSpacing);


//! Compute the volume of the solvent-excluded cavities close to each atom.
/*! Avoid storing any level-set function on a grid. Only a patch at a
 time will be used.  Solvent probes will be placed by distributing
 points on a sphere (hence the Pos in the name).  Compute the volume
 using only the sign of the distance. */
void
solventExcludedCavitiesPos(const std::vector<geom::Ball<float, 3> >& balls,
                           float probeRadius, float targetGridSpacing,
                           std::vector<float>* volumes);


} // namespace levelSet
}

#define __levelSet_solventExcludedCavitiesPos_ipp__
#include "stlib/levelSet/solventExcludedCavitiesPos.ipp"
#undef __levelSet_solventExcludedCavitiesPos_ipp__

#endif

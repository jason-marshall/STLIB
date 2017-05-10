/* -*- C++ -*- */

#ifndef __levelSet_solventExcludedCavitiesCuda_h__
#define __levelSet_solventExcludedCavitiesCuda_h__

#include "stlib/levelSet/positiveDistanceCuda.h"
#include "stlib/levelSet/powerDistanceCuda.h"
#include "stlib/levelSet/exteriorVolumeCuda.h"
#include "stlib/levelSet/gridCuda.h"
#include "stlib/levelSet/outsideCuda.h"
#include "stlib/levelSet/solventExcludedCavities.h"
#include "stlib/levelSet/solventExcludedCavitiesPosCuda.h"
#include "stlib/levelSet/dependencies.h"

#include <vector>

#include <cassert>

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
solventExcludedCavitiesPosCuda(const std::vector<geom::Ball<float, 3> >& balls,
                               float probeRadius, float targetGridSpacing);


//! Construct a level set for the solvent-excluded surface for solvents inside the protein.
void
solventExcludedCavitiesCuda(Grid<float, 3, PatchExtent>* grid,
                            const std::vector<geom::Ball<float, 3> >& balls,
                            float probeRadius);


} // namespace levelSet
}

#define __levelSet_solventExcludedCavitiesCuda_ipp__
#include "stlib/levelSet/solventExcludedCavitiesCuda.ipp"
#undef __levelSet_solventExcludedCavitiesCuda_ipp__

#endif

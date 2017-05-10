/* -*- C++ -*- */

#ifndef __levelSet_solventAccessibleCavitiesCuda_h__
#define __levelSet_solventAccessibleCavitiesCuda_h__

#include "stlib/levelSet/positiveDistanceCuda.h"
#include "stlib/levelSet/powerDistanceCuda.h"
#include "stlib/levelSet/gridCuda.h"
#include "stlib/levelSet/outsideCuda.h"
#include "stlib/levelSet/solventAccessibleCavities.h"

#include <vector>

#include <cassert>

namespace stlib
{
namespace levelSet
{


//! Compute the volume and surface area of each component of the SAC.
/*!
  \param balls The atoms that comprise the molecule.
  \param probeRadius The radius of the solvent probe.
  \param targetGridSpacing The target spacing for the AMR grid.
  \param content The vector of volumes for each component of the SAC.
  \param boundary The vector of surface areas for each component of the SAC.
*/
void
solventAccessibleCavitiesCuda(const std::vector<geom::Ball<float, 3> >& balls,
                              float probeRadius, float targetGridSpacing,
                              std::vector<float>* content,
                              std::vector<float>* boundary);


//! Construct a level set for the solvent-accessible surface for solvents inside the protein.
void
solventAccessibleCavitiesCuda(Grid<float, 3, PatchExtent>* grid,
                              const std::vector<geom::Ball<float, 3> >& balls,
                              float probeRadius);


} // namespace levelSet
}

#define __levelSet_solventAccessibleCavitiesCuda_ipp__
#include "stlib/levelSet/solventAccessibleCavitiesCuda.ipp"
#undef __levelSet_solventAccessibleCavitiesCuda_ipp__

#endif

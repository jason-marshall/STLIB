/* -*- C++ -*- */

#if !defined(__levelSet_solventExcludedCavitiesPosCuda_h__)
#define __levelSet_solventExcludedCavitiesPosCuda_h__

#include "stlib/levelSet/cuda.h"

#ifndef __CUDA_ARCH__

#include "stlib/levelSet/GridGeometry.h"
#include "stlib/cuda/check.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/numerical/constants.h"

#include <vector>

namespace stlib
{
namespace levelSet
{


//! Compute the volume of the solvent-excluded cavities.
float
solventExcludedCavitiesPosCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies);


} // namespace levelSet
}

#endif // __CUDA_ARCH__

#endif

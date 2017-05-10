/* -*- C++ -*- */

#if !defined(__levelSet_exteriorVolumeCuda_h__)
#define __levelSet_exteriorVolumeCuda_h__

#include "stlib/levelSet/cuda.h"

#ifndef __CUDA_ARCH__

#include "stlib/levelSet/GridGeometry.h"
#include "stlib/cuda/check.h"
#include "stlib/geom/kernel/Ball.h"

#include <vector>

namespace stlib
{
namespace levelSet
{


//! Compute the volume of the exterior of the union of the balls.
float
exteriorVolumeCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<bool>& isActive,
 const std::vector<geom::Ball<float, 3> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies);


} // namespace levelSet
}

#endif // __CUDA_ARCH__

#endif

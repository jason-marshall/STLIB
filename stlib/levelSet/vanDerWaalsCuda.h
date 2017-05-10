/* -*- C++ -*- */

#if !defined(__levelSet_vanDerWaalsCuda_h__)
#define __levelSet_vanDerWaalsCuda_h__

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


//! Compute the volume for the van der Waals domain.
/*! Do not construct an (AMR) Grid to store the level set. For each block,
  only work with one patch at a time. */
float
vanDerWaalsCuda(const std::vector<geom::Ball<float, 3> >& balls,
                float targetGridSpacing);

//! Compute the volume of the solvent-excluded cavities.
float
vanDerWaalsCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies);


} // namespace levelSet
}

#endif // __CUDA_ARCH__

#endif

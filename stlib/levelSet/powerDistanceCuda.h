/* -*- C++ -*- */

#if !defined(__levelSet_powerDistanceCuda_h__)
#define __levelSet_powerDistanceCuda_h__

#ifndef __CUDA_ARCH__

#include "stlib/levelSet/cuda.h"
#include "stlib/levelSet/Grid.h"
#include "stlib/geom/kernel/BallSquared.h"
#include "stlib/geom/kernel/Ball.h"

#include <vector>

namespace stlib
{
namespace levelSet
{

//! Construct a level set for the power distance to a set of balls.
/*! For each grid point, compute the distance to each of the balls.
  Translate the data to CUDA format and call powerDistanceKernel(). */
void
powerDistanceCuda(container::EquilateralArray<float, 3, PatchExtent>* patch,
                  const std::array<float, 3>& lowerCorner,
                  float spacing,
                  const std::vector<geom::BallSquared<float, 3> >& balls);

//! Construct a level set for the power distance to a set of balls.
/*! The dependencies have been computed and the grid has already been
  refined. */
void
negativePowerDistanceCuda
(Grid<float, 3, PatchExtent>* grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies,
 float maxDistance = std::numeric_limits<float>::infinity());

//! Construct a level set for the power distance to a set of balls.
void
negativePowerDistanceCuda(Grid<float, 3, PatchExtent>* grid,
                          const std::vector<geom::Ball<float, 3> >& balls);

} // namespace levelSet
}

#endif

#endif

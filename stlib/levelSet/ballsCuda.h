/* -*- C++ -*- */

#ifndef __levelSet_ballsCuda_h__
#define __levelSet_ballsCuda_h__

#include "stlib/levelSet/cuda.h"
#include "stlib/levelSet/Grid.h"
#include "stlib/cuda/check.h"

namespace stlib
{
namespace levelSet
{

//! Allocate device memory for the ball indices.
void
allocateBallIndicesCuda
(const container::StaticArrayOfArrays<unsigned>& dependencies,
 unsigned** ballIndexOffsetsDev,
 unsigned** packedBallIndicesDev);

//! Allocate device memory for the ball indices.
/*! Refined patches may have empty sets of dependencies. */
void
allocateBallIndicesCuda
(const Grid<float, 3, PatchExtent>& grid,
 const container::StaticArrayOfArrays<unsigned>& dependencies,
 unsigned** ballIndexOffsetsDev,
 unsigned** packedBallIndicesDev);



} // namespace levelSet
}

#define __levelSet_ballsCuda_ipp__
#include "stlib/levelSet/ballsCuda.ipp"
#undef __levelSet_ballsCuda_ipp__

#endif

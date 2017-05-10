/* -*- C++ -*- */

#if !defined(__levelSet_outsideCuda_h__)
#define __levelSet_outsideCuda_h__

#ifndef __CUDA_ARCH__

#include "stlib/ext/array.h"

#include <vector>

namespace stlib
{
namespace levelSet
{

//! Mark the outside grid points as negative infinity.
/*!
  The input grid is a level set function. Negative/positive distance denote
  the interior/exterior of the object. Points on the boundary with positive
  distance are defined to be outside. Any grid point with positive distance
  that is a neighbor of an outside point is also outside. This criterion
  allows us to distinguish between domains that are outside the object and
  domains that are cavities.
*/
void
markOutsideAsNegativeInf
(const std::array<std::size_t, 3>& gridExtents,
 std::size_t numRefined,
 float* patchesDev,
 const uint3* indicesDev,
 const std::vector<std::array<std::size_t, 3> >& negativePatches,
 std::vector<bool>* outsideAtLowerCorners);

} // namespace levelSet
}

#endif

#endif

/* -*- C++ -*- */

#ifndef __levelSet_gridCuda_h__
#define __levelSet_gridCuda_h__

#include "stlib/levelSet/Grid.h"
#include "stlib/levelSet/cuda.h"
#include "stlib/cuda/check.h"
#include "stlib/container/SimpleMultiIndexExtentsIterator.h"

namespace stlib
{
namespace levelSet
{


//! Allocate device memory for the active patch indices.
void
allocateGridIndicesCuda(const GridGeometry<3, PatchExtent, float>& grid,
                        std::size_t numRefined,
                        const std::vector<bool>& isActive,
                        uint3** indicesDev);

//! Allocate device memory for the grid.
void
allocateGridCuda(const Grid<float, 3, PatchExtent>& grid, float** patchesDev,
                 uint3** indicesDev);

} // namespace levelSet
}

#define __levelSet_gridCuda_ipp__
#include "stlib/levelSet/gridCuda.ipp"
#undef __levelSet_gridCuda_ipp__

#endif

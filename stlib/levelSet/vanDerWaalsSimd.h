// -*- C++ -*-

#if !defined(__levelSet_vanDerWaalsSimd_h__)
#define __levelSet_vanDerWaalsSimd_h__

// Get the patch extent.
#include "stlib/levelSet/cuda.h"
#include "stlib/levelSet/GridGeometry.h"
#include "stlib/levelSet/PatchActive.h"

#include "stlib/container/SimpleMultiIndexExtentsIterator.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/numerical/constants.h"

#include <vector>

namespace stlib
{
namespace levelSet
{


//! Compute the volume and surface area for the van der Waals domain.
/*! Construct an (AMR) Grid to store the level set. Compute the power
  distance to determine the level set. */
float
vanDerWaalsSimd(const std::vector<geom::Ball<float, 3> >& balls,
                float targetGridSpacing);

//! Compute the volume of the solvent-excluded cavities.
float
vanDerWaalsSimd
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies);


} // namespace levelSet
}

#define __levelSet_vanDerWaalsSimd_ipp__
#include "stlib/levelSet/vanDerWaalsSimd.ipp"
#undef __levelSet_vanDerWaalsSimd_ipp__

#endif

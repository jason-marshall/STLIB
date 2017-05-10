// -*- C++ -*-

#if !defined(__levelSet_vanDerWaals_h__)
#define __levelSet_vanDerWaals_h__

#include "stlib/levelSet/powerDistance.h"
#include "stlib/levelSet/marchingSimplices.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetVanDerWaals van der Waals Surface
*/
//@{


//! Compute the volume and surface area for the van der Waals domain.
/*! Construct an (AMR) Grid to store the level set. Compute the power
  distance to determine the level set. */
template<typename _T, std::size_t _D>
std::pair<_T, _T>
vanDerWaals(const std::vector<geom::Ball<_T, _D> >& balls,
            _T targetGridSpacing);


//@}

} // namespace levelSet
}

#define __levelSet_vanDerWaals_ipp__
#include "stlib/levelSet/vanDerWaals.ipp"
#undef __levelSet_vanDerWaals_ipp__

#endif

// -*- C++ -*-

#if !defined(__levelSet_signedDistance_h__)
#define __levelSet_signedDistance_h__

#include "stlib/levelSet/negativeDistance.h"
#include "stlib/levelSet/positiveDistance.h"

namespace stlib
{
namespace levelSet
{


/*! \defgroup levelSetSignedDistance Signed Distance
These functions calculate the signed distance for the union of a set of balls.
*/
//@{

//! Compute the signed distance for a union of balls.
/*!
  Negative distances are correctly computed up to the minimum radius of the set
  of balls. The positive distance is computed up to the specified distance.
  Set the far-away distances to \f$\pm \infty\f$.
*/
template<typename _T, std::size_t _D>
void
signedDistance(GridUniform<_T, _D>* grid,
               const std::vector<geom::Ball<_T, _D> >& balls,
               _T maxDistance);


//@}

} // namespace levelSet
}

#define __levelSet_signedDistance_ipp__
#include "stlib/levelSet/signedDistance.ipp"
#undef __levelSet_signedDistance_ipp__

#endif

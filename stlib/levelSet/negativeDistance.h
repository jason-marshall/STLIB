// -*- C++ -*-

#if !defined(__levelSet_negativeDistance_h__)
#define __levelSet_negativeDistance_h__

#include "stlib/levelSet/GridUniform.h"
#include "stlib/levelSet/Grid.h"
#include "stlib/levelSet/geometry.h"
#include "stlib/levelSet/IntersectionPoint.h"
#include "stlib/levelSet/IntersectionCircle.h"

#include "stlib/geom/grid/SimpleRegularGrid.h"

namespace stlib
{
namespace levelSet
{


/*! \defgroup levelSetNegativeDistance Negative Distance
These functions calculate the Negative distance for the union of a set of balls.
*/
//@{

//! Compute the negative distances for a union of balls.
/*!
  The distance is correctly computed up to the minimum radius of the set
  of balls. Set the positive distances to NaN. Set the negative far-away
  distances to \f$-\infty\f$.
*/
template<typename _T, std::size_t _D>
void
negativeDistance(GridUniform<_T, _D>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls);


//! Compute the negative distances for a union of balls.
/*!
  The distance is correctly computed up to the minimum radius of the set
  of balls. Set the positive distances to NaN. Set the negative far-away
  distances to \f$-\infty\f$.
*/
template<typename _T, std::size_t _D, std::size_t N>
void
negativeDistance(Grid<_T, _D, N>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls);


//@}

} // namespace levelSet
}

#define __levelSet_negativeDistance_ipp__
#include "stlib/levelSet/negativeDistance.ipp"
#undef __levelSet_negativeDistance_ipp__

#endif

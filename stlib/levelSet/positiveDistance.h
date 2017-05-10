// -*- C++ -*-

#if !defined(__levelSet_positiveDistance_h__)
#define __levelSet_positiveDistance_h__

#include "stlib/levelSet/GridUniform.h"
#include "stlib/levelSet/Grid.h"

#include "stlib/geom/grid/SimpleRegularGrid.h"
#include "stlib/geom/kernel/Ball.h"

namespace stlib
{
namespace levelSet
{


/*! \defgroup levelSetPositiveDistance Positive Distance
These functions calculate the positive distance for the union of a set of balls.
*/
//@{

//! Construct a level set for a union of balls.
/*!
  The level set function is the positive distance from the surface for points
  outside of the union of balls. For points inside the balls the function
  has negative values, but due to intersecting characteristics it is not
  necessarily the signed distance.
*/
template<typename _T, std::size_t _D>
void
positiveDistance(container::SimpleMultiArrayRef<_T, _D>* grid,
                 const geom::BBox<_T, _D>& domain,
                 const std::vector<geom::Ball<_T, _D> >& balls,
                 _T offset = 0, _T maxDistance = 0);


//! Construct a level set for a union of balls.
/*!
  The level set function is the positive distance from the surface for points
  outside of the union of balls. For points inside the balls the function
  has negative values, but due to intersecting characteristics it is not
  necessarily the signed distance.
*/
template<typename _T, std::size_t _D>
inline
void
positiveDistance(GridUniform<_T, _D>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls,
                 _T offset = 0, _T maxDistance = 0)
{
  positiveDistance(grid, grid->domain(), balls, offset, maxDistance);
}


//! Construct a level set for a union of balls.
/*!
  The level set function is the positive distance from the surface for points
  outside of the union of balls. For points inside the balls the function
  has negative values, but due to intersecting characteristics it is not
  necessarily the signed distance.
*/
template<typename _T, std::size_t _D, std::size_t N>
void
positiveDistance(Grid<_T, _D, N>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls,
                 _T offset = 0, _T maxDistance = 0);


//! Construct the level set function for the union of the balls.
/*!
  \param grid The grid on which the level set function will be calculated.
  \param balls The input balls. Note that we pass this parameter by value
  as we will modify in the radii in the internal calculations.
  \param maxDistance How far to calculate the distance past the union of
  the offset balls.

  Only the positive distance is guaranteed to be computed. Patches with
  all negative distances will be unrefined and have a fill value of
  negative infinity. Patches with far away positive distance will be unrefined
  and have a fill value of NaN.
*/
template<typename _T, std::size_t _D, std::size_t N>
inline
void
positiveDistanceOutside(Grid<_T, _D, N>* grid,
                        std::vector<geom::Ball<_T, _D> > balls, _T maxDistance);

//@}

} // namespace levelSet
}

#define __levelSet_positiveDistance_ipp__
#include "stlib/levelSet/positiveDistance.ipp"
#undef __levelSet_positiveDistance_ipp__

#endif

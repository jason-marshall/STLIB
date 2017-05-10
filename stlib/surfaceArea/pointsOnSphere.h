// -*- C++ -*-

/*!
  \file surfaceArea/kernel/pointsOnSphere.h
  \brief Distribute points on a sphere.
*/
/*!
  \page surfaceArea_kernel_pointsOnSphere Distribute Points on a Sphere

  CONTINUE.
*/

#if !defined(__surfaceArea_kernel_pointsOnSphere_h__)
#define __surfaceArea_kernel_pointsOnSphere_h__

#include "stlib/numerical/constants.h"

#include <iterator>

#include <cstddef>
#include <cmath>

namespace stlib
{
namespace surfaceArea
{

//-----------------------------------------------------------------------------
/*! \defgroup surfaceArea_kernel_pointsOnSphere Distribute Points on a Sphere */
// @{


//! Distribute points on a sphere with the golden section spiral algorithm.
/*!
  http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere
*/
template<typename _Point, typename _OutputIterator>
void
distributePointsOnSphereWithGoldenSectionSpiral(std::size_t size,
    _OutputIterator points);

// @}

} // namespace surfaceArea
}

#define __surfaceArea_kernel_pointsOnSphere_ipp__
#include "stlib/surfaceArea/pointsOnSphere.ipp"
#undef __surfaceArea_kernel_pointsOnSphere_ipp__

#endif

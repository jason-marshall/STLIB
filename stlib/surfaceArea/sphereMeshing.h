// -*- C++ -*-

/*!
  \file surfaceArea/sphereMeshing.h
  \brief Calculate solvent-accessible surface areas considering pairwise interactions.
*/

#if !defined(__surfaceArea_sphereMeshing_h__)
#define __surfaceArea_sphereMeshing_h__

#include "stlib/surfaceArea/pointsOnSphere.h"

#include "stlib/geom/kernel/BBox.h"
#include "stlib/ext/vector.h"

#include <vector>

namespace stlib
{
namespace surfaceArea
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;

//! Clip the points on a sphere with another sphere.
/*! The input is a set of points along with the number of active points.
 The active points are in the range [0..numActive). The inactive points
 are in the range [numActive..points->size()). The active points are
 clipped by a sphere (defined by the specified center and radius).
 Clipped points are moved into the inactive range. The number of active
 points after clipping is returned. */
template<typename Float>
std::size_t
clip(std::vector<std::array<Float, 3> >* points,
     std::size_t numActive, const std::array<Float, 3>& center,
     const Float radius);

} // namespace surfaceArea
}

#define __surfaceArea_sphereMeshing_ipp__
#include "stlib/surfaceArea/sphereMeshing.ipp"
#undef __surfaceArea_sphereMeshing_ipp__

#endif

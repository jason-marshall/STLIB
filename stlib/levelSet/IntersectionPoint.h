// -*- C++ -*-

#if !defined(__levelSet_IntersectionPoint_h__)
#define __levelSet_IntersectionPoint_h__

#include "stlib/geom/kernel/Ball.h"
#include "stlib/geom/kernel/Point.h"

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! A point with normal on a surface.
/*!
  \param _T is the number type.
  \param _D is the dimension.
*/
template<typename _T, std::size_t _D>
struct IntersectionPoint {
  //! The location of the point.
  std::array<_T, _D> location;
  //! The outward normal to the surface.
  std::array<_T, _D> normal;
  //! How far to compute the distance.
  _T radius;
};


} // namespace levelSet
}

#define __levelSet_IntersectionPoint_ipp__
#include "stlib/levelSet/IntersectionPoint.ipp"
#undef __levelSet_IntersectionPoint_ipp__

#endif

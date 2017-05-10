// -*- C++ -*-

/*!
  \file tesselate_sphere_unindexed.h
  \brief Function for tesselating a unit sphere.
*/

#if !defined(__tesselate_sphere_unindexed_h__)
#define __tesselate_sphere_unindexed_h__

#include "stlib/geom/kernel/Point.h"

#include <list>

#include <cmath>
#include <cassert>

namespace stlib
{
namespace mst
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

} // namespace mst
}

#define __tesselate_sphere_unindexed_ipp__
#include "stlib/mst/tesselate_sphere_unindexed.ipp"
#undef __tesselate_sphere_unindexed_ipp__

#endif

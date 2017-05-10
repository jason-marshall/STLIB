// -*- C++ -*-

/*!
  \file geom/mesh/iss/closestSimplex.h
  \brief For a set of points, determine the closest simplex in a mesh.
*/

#if !defined(__geom_mesh_iss_closestSimplex_h__)
#define __geom_mesh_iss_closestSimplex_h__

#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/simplex/simplex_distance.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/ads/functor/Dereference.h"

namespace stlib
{
namespace geom {


//! For a set of points, determine the closest simplex in a mesh.
/*!
  Of course, this algorithm works for arbitrary points and meshes. However, it
  is only efficient when most of the points are contained in the mesh.
*/
template<template<std::size_t, typename> class _Orq, std::size_t SpaceD,
         typename _T>
void
closestSimplex(const IndSimpSet<SpaceD, SpaceD, _T>& mesh,
               const std::vector<std::array<_T, SpaceD> >& points,
               std::vector<std::size_t>* indices);


} // namespace geom
}

#define __geom_mesh_iss_closestSimplex_ipp__
#include "stlib/geom/mesh/iss/closestSimplex.ipp"
#undef __geom_mesh_iss_closestSimplex_ipp__

#endif

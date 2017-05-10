// -*- C++ -*-

/*!
  \file geom/mesh/iss/penetration.h
  \brief Report penetrations.
*/

#if !defined(__geom_mesh_iss_penetration_h__)
#define __geom_mesh_iss_penetration_h__

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/set.h"

#include "stlib/geom/orq/CellArrayStatic.h"

#include <tuple>

#include <map>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_penetration Report penetrations.
*/
//@{

//! Report the points that penetrate the solid mesh.
/*!
  \relates IndSimpSetIncAdj

  \param mesh The solid mesh.
  \param pointsBeginning The beginning of a list of points.
  \param pointsEnd The end of a list of points.
  \param penetrations Output iterator for the penetrations.
  \return The number of points that penetrate the solid.

  Penetrations are reported as a 3-tuple: point index, mesh simplex index, and
  closest point on the mesh boundary.  The structure for this is
  std::tuple<std::size_t, std::size_t, std::array<T,3> > .
*/
template < std::size_t N, typename T,
         typename PointRandomAccessIterator,
         typename TupleOutputIterator >
inline
std::size_t
reportPenetrations(const IndSimpSetIncAdj<N, N, T>& mesh,
                   PointRandomAccessIterator pointsBeginning,
                   PointRandomAccessIterator pointsEnd,
                   TupleOutputIterator penetrations);

//! Return the maximum incident edge length.
template<std::size_t N, typename T>
T
maximumIncidentEdgeLength(const IndSimpSetIncAdj<N, N, T>& mesh, std::size_t n);

//! Report the maximum relative penetration for a boundary node.
/*!
  \relates IndSimpSetIncAdj

  \param mesh The solid mesh.
  \return The maximum relative penetration.

  Let the mesh have C components. We consider penetrations of boundary nodes
  from one component into another component. We do not consider possible
  self-penetrations within a single component. The relative penetration
  of a node is the penetration distance divided by the maximum incident edge
  length.
*/
template<std::size_t N, typename T>
T
maximumRelativePenetration(const IndSimpSetIncAdj<N, N, T>& mesh);

//@}

} // namespace geom
}

#define __geom_mesh_iss_penetration_ipp__
#include "stlib/geom/mesh/iss/penetration.ipp"
#undef __geom_mesh_iss_penetration_ipp__

#endif

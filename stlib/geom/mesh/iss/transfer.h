// -*- C++ -*-

/*!
  \file geom/mesh/iss/transfer.h
  \brief Transfer fields for indexed simplex sets.
*/

#if !defined(__geom_mesh_iss_transfer_h__)
#define __geom_mesh_iss_transfer_h__

#include "stlib/geom/mesh/iss/ISS_Interpolate.h"
#include "stlib/geom/mesh/iss/IndSimpSet.h"

#include <iostream>
#include <string>

#include <cassert>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_transfer Transfer Fields
  These functions are used to transfer fields from one simplicial mesh to
  another.
*/
//@{

//! Determine the simplex indices in an ISS for transfering fields.
/*!
  \relates IndSimpSet

  Determine the simplex indices in the source mesh that should be used
  for transfering fields to the specified points.

  \param mesh is the simplicial mesh.
  \param points are the interpolation points.
  \param indices are indices of simplices in the mesh that should be
  used in transfering fields.  The \c indices array must be the same size
  as the \c points array.

  The template parameters can be deduced from the arguments.
  - \c ISS is the indexed simplex set type.  It can be an IndSimpSet or
  an IndSimpSetIncAdj.
  - \c PointArray is an ADS array of points.
  - \c IndexArray is an ADS array of indices.
*/
template < std::size_t N, std::size_t M, typename T,
         class PointArray, class IndexArray >
void
transferIndices(const IndSimpSet<N, M, T>& mesh,
                const PointArray& points,
                IndexArray* indices);


//! Transfer fields for indexed simplex sets.
/*!
  \relates IndSimpSet

  Use linear interpolation to transfer the fields from the source mesh to
  the target vertices.

  \param mesh The simplicial mesh.
  \param sourceFields are the field values at the vertices of the mesh.
  \param points are the interpolation points.
  \param targetFields are the field values at the interpolation points.

  The template parameters can be deduced from the arguments.
  - \c ISS is the indexed simplex set type.  It can be an IndSimpSet or
  an IndSimpSetIncAdj.
  - \c SourceFieldArray is an ADS array of the source fields.
  - \c PointArray is an ADS array of points.
  - \c TargetFieldArray is an ADS array of the target fields.
*/
template < std::size_t N, std::size_t M, typename T,
         class SourceFieldArray, class PointArray, class TargetFieldArray >
void
transfer(const IndSimpSet<N, M, T>& mesh,
         const SourceFieldArray& sourceFields,
         const PointArray& points,
         TargetFieldArray* targetFields);

//@}

} // namespace geom
}

#define __geom_mesh_iss_transfer_ipp__
#include "stlib/geom/mesh/iss/transfer.ipp"
#undef __geom_mesh_iss_transfer_ipp__

#endif

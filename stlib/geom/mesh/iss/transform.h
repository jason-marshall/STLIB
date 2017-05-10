// -*- C++ -*-

/*!
  \file geom/mesh/iss/transform.h
  \brief Implements operations that transform a IndSimpSet.
*/

#if !defined(__geom_mesh_iss_transform_h__)
#define __geom_mesh_iss_transform_h__

#include "stlib/geom/mesh/iss/boundaryCondition.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/set.h"

#include "stlib/geom/mesh/simplex/SimplexJac.h"

#include "stlib/ads/iterator/IntIterator.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/ads/iterator/TransformIterator.h"

#include <stack>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_transform Transform Vertices or Simplices
  These are function that transform indexed simplex sets.
*/
//@{

//! Pack the ISS to get rid of unused vertices.
/*!
  \relates IndSimpSet

  Adjust the vertex indices accordingly.
*/
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
void
pack(IndSimpSet<N, M, T>* mesh, IntOutputIterator usedVertexIndices);



//! Pack the ISS to get rid of unused vertices.
/*!
  \relates IndSimpSet

  Adjust the vertex indices accordingly.
*/
template<std::size_t N, std::size_t M, typename T>
inline
void
pack(IndSimpSet<N, M, T>* mesh) {
   pack(mesh, ads::TrivialOutputIterator());
}


//! Orient each simplex so it has non-negative volume.
/*! \relates IndSimpSet */
template<std::size_t N, typename T>
void
orientPositive(IndSimpSet<N, N, T>* mesh);


//! Reverse the orientation of each simplex.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
reverseOrientation(IndSimpSet<N, M, T>* mesh);


//! Transform each vertex in the range with the specified function.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class UnaryFunction >
void
transform(IndSimpSet<N, M, T>* mesh,
          IntForIter beginning, IntForIter end, const UnaryFunction& f);


//! Transform each vertex in the mesh with the specified function.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         class UnaryFunction >
void
transform(IndSimpSet<N, M, T>* mesh, const UnaryFunction& f);


//! Transform each vertex in the range with the closest point in the normal direction.
/*! \relates IndSimpSetIncAdj */
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class ISS >
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          IntForIter beginning, IntForIter end,
          const ISS_SD_ClosestPointDirection<ISS>& f);


//! Transform each vertex in the mesh with the closest point in the normal direction.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T, class ISS>
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          const ISS_SD_ClosestPointDirection<ISS>& f);


//! Transform each vertex in the range with the closer point in the normal direction.
/*! \relates IndSimpSetIncAdj */
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class ISS >
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          IntForIter beginning, IntForIter end,
          const ISS_SD_CloserPointDirection<ISS>& f);


//! Transform each vertex in the mesh with the closer point in the normal direction.
/*! \relates IndSimpSetIncAdj */
template < std::size_t N, std::size_t M, typename T,
         class ISS >
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          const ISS_SD_CloserPointDirection<ISS>& f);


//! Remove simplices until there are none with minimum adjacencies less than specified.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
void
removeLowAdjacencies(IndSimpSetIncAdj<N, M, T>* mesh,
                     std::size_t minRequiredAdjencies);


//@}

} // namespace geom
}

#define __geom_mesh_iss_transform_ipp__
#include "stlib/geom/mesh/iss/transform.ipp"
#undef __geom_mesh_iss_transform_ipp__

#endif

 // -*- C++ -*-

/*!
  \file geom/mesh/iss/equality.h
  \brief Equality operators for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_equality_h__)
#define __geom_mesh_iss_equality_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_equality Equality
  Test the equality of two indexed simplex sets.  These functions are for
  debugging purposes only.  They don't do anything fancy like check if the
  vertices or simplices are given in different order.
*/
// @{

//! Return true if the vertices and indexed simplices are equal.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M,
         typename T >
inline
bool
operator==(const IndSimpSet<N, M, T>& x,
           const IndSimpSet<N, M, T>& y) {
   return (x.vertices == y.vertices &&
           x.indexedSimplices == y.indexedSimplices);
}

//! Return true if the vertices and indexed simplices are not equal.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M,
         typename T >
inline
bool
operator!=(const IndSimpSet<N, M, T>& x,
           const IndSimpSet<N, M, T>& y) {
   return !(x == y);
}

//
// Equality.
//

//! Return true if the meshes are equal.
/*! \relates IndSimpSetIncAdj */
template < std::size_t N, std::size_t M,
         typename T >
inline
bool
operator==(const IndSimpSetIncAdj<N, M, T>& x,
           const IndSimpSetIncAdj<N, M, T>& y) {
   return (x.vertices == y.vertices &&
           x.indexedSimplices == y.indexedSimplices &&
           x.incident == y.incident &&
           x.adjacent == y.adjacent);
}

//! Return true if the meshes are not equal.
/*! \relates IndSimpSetIncAdj */
template < std::size_t N, std::size_t M,
         typename T >
inline
bool
operator!=(const IndSimpSetIncAdj<N, M, T>& x,
           const IndSimpSetIncAdj<N, M, T>& y) {
   return !(x == y);
}


// @}

} // namespace geom
}

#endif

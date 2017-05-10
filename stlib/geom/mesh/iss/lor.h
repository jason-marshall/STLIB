// -*- C++ -*-

/*!
  \file geom/mesh/iss/lor.h
  \brief Reorder for good locality of reference.
*/

#if !defined(__geom_mesh_iss_lor_h__)
#define __geom_mesh_iss_lor_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/accessors.h"

#include "stlib/lorg/order.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_lor Locality of Reference
*/
//@{


//! Reorder the vertices and simplices for good locality of reference.
/*!
  \relates IndSimpSet

  \param mesh Pointer to the simplicial mesh.
*/
template<typename _Integer, std::size_t N, std::size_t M, typename T>
void
mortonOrder(IndSimpSet<N, M, T>* mesh);


//! Reorder the vertices and simplices for good locality of reference.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
*/
template<typename _Integer, std::size_t N, std::size_t M, typename T>
void
mortonOrder(IndSimpSetIncAdj<N, M, T>* mesh);


//! Sort along the axis of greatest extent.
/*!
  \relates IndSimpSet

  \param mesh Pointer to the simplicial mesh.
*/
template<std::size_t N, std::size_t M, typename T>
void
axisOrder(IndSimpSet<N, M, T>* mesh);


//! Sort along the axis of greatest extent.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
*/
template<std::size_t N, std::size_t M, typename T>
void
axisOrder(IndSimpSetIncAdj<N, M, T>* mesh);


//! Put the vertices and simplices in a random order.
/*!
  \relates IndSimpSet

  \param mesh Pointer to the simplicial mesh.
*/
template<std::size_t N, std::size_t M, typename T>
void
randomOrder(IndSimpSet<N, M, T>* mesh);


//! Put the vertices and simplices in a random order.
/*!
  \relates IndSimpSetIncAdj

  \param mesh Pointer to the simplicial mesh.
*/
template<std::size_t N, std::size_t M, typename T>
void
randomOrder(IndSimpSetIncAdj<N, M, T>* mesh);


//@}

} // namespace geom
}

#define __geom_mesh_iss_lor_ipp__
#include "stlib/geom/mesh/iss/lor.ipp"
#undef __geom_mesh_iss_lor_ipp__

#endif

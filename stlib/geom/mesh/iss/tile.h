// -*- C++ -*-

/*!
  \file geom/mesh/iss/tile.h
  \brief Tile a rectilinear region with simplices.
*/

#if !defined(__geom_mesh_iss_tile_h__)
#define __geom_mesh_iss_tile_h__

#include "stlib/geom/mesh/iss/buildFromSimplices.h"
#include "stlib/geom/mesh/iss/transform.h"

#include "stlib/ads/functor/constant.h"

#include <iostream>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_tile Tiling
  These functions use the cookie-cutter algorithm to mesh an object.
  The object is described implicitly with a level set function.

  Future work: Tile inside and around the object.  Refine the mesh
  near the boundary.  Then apply the cookie cutter algorithm.  This will
  increase fidelity at the boundary while maintaining the desired edge
  length in the interior.
*/
//@{

//! Tile the object with equilateral triangles.
/*!
  \relates IndSimpSet

  \param domain is the rectangular domain that contains the object.
  \param length is the triangle edge length.
  \param f is the level set description of the object.
  \param mesh is the indexed simplex set.

  The template parameters can be deduced from the arguments.
  - \c T is the number type.
  - \c LSF is the level set function that describes the object.  Negative
    values are inside.
*/
template < typename T,  // number type
         class LSF >   // Level Set Function
void
tile(const BBox<T, 2>& domain, T length, const LSF& f,
     IndSimpSet<2, 2, T>* mesh);


//! Tile the rectangular region with equilateral triangles.
/*!
  \relates IndSimpSet

  \param domain is the rectangular domain to tile.
  \param length is the triangle edge length.
  \param mesh is the indexed simplex set.

  The template parameters can be deduced from the arguments.
  - \c T is the number type.

  \note This function simply calls the above tiling function with a trivial
  level set function.
*/
template<typename T>    // number type
inline
void
tile(const BBox<T, 2>& domain, const T length,
     IndSimpSet<2, 2, T>* mesh) {
   typedef typename IndSimpSet<2, 2, T>::Vertex Vertex;
   tile(domain, length, ads::constructUnaryConstant<Vertex>(-1.0), mesh);
}




//! Tile the object with a body-centered cubic lattice.
/*!
  \relates IndSimpSet

  \param domain is the rectilinear domain to tile.
  \param length is the maximum tetrahedron edge length.
  \param f is the level set description of the object.
  \param mesh is the indexed simplex set.

  \image html bcc0.jpg "A BCC block with 12 tetrahedra."
  \image latex bcc0.jpg "A BCC block with 12 tetrahedra." width=0.5\textwidth

  The template parameters can be deduced from the arguments.
  - \c T is the number type.
  - \c LSF is the level set function that describes the object.  Negative
    values are inside.
*/
template < typename T,  // number type
         class LSF >   // Level Set Function
void
tile(const BBox<T, 3>& domain, const T length, const LSF& f,
     IndSimpSet<3, 3, T>* mesh);


//! Tile the rectilinear region with a body-centered cubic lattice.
/*!
  \relates IndSimpSet

  \param domain is the rectilinear domain to tile.
  \param length is the maximum tetrahedron edge length.
  \param mesh is the indexed simplex set.

  The template parameters can be deduced from the arguments.
  - \c T is the number type.

  \note This function simply calls the above tiling function with a trivial
  level set function.
*/
template<typename T>    // number type
inline
void
tile(const BBox<T, 3>& domain, const T length,
     IndSimpSet<3, 3, T>* mesh) {
   typedef typename IndSimpSet<3, 3, T>::Vertex Vertex;
   tile(domain, length, ads::constructUnaryConstant<Vertex>(-1.0), mesh);
}

//@}

} // namespace geom
}

#define __geom_mesh_iss_tile_ipp__
#include "stlib/geom/mesh/iss/tile.ipp"
#undef __geom_mesh_iss_tile_ipp__

#endif

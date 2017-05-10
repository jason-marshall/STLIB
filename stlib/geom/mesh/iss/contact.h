// -*- C++ -*-

/*!
  \file geom/mesh/iss/contact.h
  \brief Remove contact.
*/

#if !defined(__geom_mesh_iss_contact_h__)
#define __geom_mesh_iss_contact_h__

#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_contact Remove contact.
*/
//@{


//! Move the vertices to remove contact.
/*!
  \param surface The surface of an object.
  \param verticesBeginning The beginning of a range of vertices.
  \param verticesEnd The end of a range of vertices.
  \return The number of vertices moved.

  For each vertex in the range: If the vertex is inside the object, move it
  to lie on the surface of the object (specifically, the closest point on
  the surface).

  One can use this function to roughly remove contact between two solid meshes.
  Consider two tetrahedron meshes A and B.  Extract the boundary of A and
  the boundary vertices of B.  Then call this function with the triangle mesh
  and the boundary vertices.  Update the position of the boundary vertices
  of B.  Now the boundary vertices of B do not penetrate A.  (B itself may
  stil intersect A, though.) Next repeat these steps with the roles of A and
  B reversed.  Now the boundary vertices of A do not penetrate B.

  Note that since we have change the boundary of A, it is possible that some
  boundary vertices of B now penetrate A.  (It's not likely, but it's
  possible.)  If it is necessary that this not happen, keep calling this
  function until no vertices are moved.


  \relates IndSimpSet
*/
template < std::size_t N, typename T,
         typename VertexForwardIterator >
std::size_t
removeContact(const IndSimpSet < N, N - 1, T > & surface,
              VertexForwardIterator verticesBeginning,
              VertexForwardIterator verticesEnd);


//@}

} // namespace geom
}

#define __geom_mesh_iss_contact_ipp__
#include "stlib/geom/mesh/iss/contact.ipp"
#undef __geom_mesh_iss_contact_ipp__

#endif

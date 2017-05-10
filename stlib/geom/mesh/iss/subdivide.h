// -*- C++ -*-

/*!
  \file geom/mesh/iss/subdivide.h
  \brief Subdivision (Uniform in 1-D and 2-D.)
*/

#if !defined(__geom_mesh_iss_subdivide_h__)
#define __geom_mesh_iss_subdivide_h__

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/container/TriangularArray.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_subdivide Subdivision for IndSimpSet
*/
//@{

//! Subdivide by splitting each simplex in half.
/*! \relates IndSimpSet */
template<std::size_t N, typename T>
void
subdivide(const IndSimpSet<N, 1, T>& in, IndSimpSet<N, 1, T>* out);


//! Subdivide by splitting each simplex into four similar simplices.
/*! \relates IndSimpSet */
template<std::size_t N, typename T>
void
subdivide(const IndSimpSetIncAdj<N, 2, T>& in, IndSimpSet<N, 2, T>* out);


//! Subdivide by splitting each simplex into eight simplices.
/*! \relates IndSimpSet */
template<std::size_t N, typename T>
void
subdivide(const IndSimpSetIncAdj<N, 3, T>& in, IndSimpSet<N, 3, T>* out);

//@}

} // namespace geom
}

#define __geom_mesh_iss_subdivide_ipp__
#include "stlib/geom/mesh/iss/subdivide.ipp"
#undef __geom_mesh_iss_subdivide_ipp__

#endif

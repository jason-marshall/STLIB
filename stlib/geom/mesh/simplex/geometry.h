// -*- C++ -*-

/*!
  \file geom/mesh/simplex/geometry.h
  \brief Geometric functions for simplices.
*/

#if !defined(__geom_mesh_simplex_geometry_h__)
#define __geom_mesh_simplex_geometry_h__

#include "stlib/geom/kernel/simplexTopology.h"

#include "stlib/geom/kernel/BBox.h"
#include "stlib/geom/kernel/Hyperplane.h"

#include "stlib/numerical/constants.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup simplex_geometry Geometric Functions
*/
//@{

//! Calculate a bounding box around the simplex.
template<typename _T, std::size_t N, typename NumberT>
inline
void
computeBBox(const std::array<_T, N>& simplex,
            BBox<NumberT, N - 1>* bb) {
  *bb = specificBBox<BBox<NumberT, N - 1> >(simplex.begin(), simplex.end());
}

//! Calculate the centroid of the simplex.
template<typename _T, std::size_t N>
inline
void
computeCentroid(const std::array<_T, N>& simplex, _T* centroid) {
   typedef typename _T::value_type Number;
   std::fill(centroid->begin(), centroid->end(), 0);
   for (std::size_t i = 0; i != simplex.size(); ++i) {
      *centroid += simplex[i];
   }
   *centroid /= Number(simplex.size());
}

//---------------------------------------------------------------------------
// Angles
//---------------------------------------------------------------------------

//! The dihedral angle between two faces.
template<typename _T>
_T
computeAngle(const std::array < std::array<_T, 3>, 3 + 1 > & s, std::size_t a,
             std::size_t b);


//! The solid angle at a vertex.
template<typename _T>
_T
computeAngle(const std::array < std::array<_T, 3>, 3 + 1 > & s,
             std::size_t n);

//! The interior angle at a vertex.
template<typename _T>
_T
computeAngle(const std::array < std::array<_T, 2>, 2 + 1 > & s,
             std::size_t n);

//! The interior angle at a vertex is 1.
template<typename _T>
_T
computeAngle(const std::array < std::array<_T, 1>, 1 + 1 > & s,
             std::size_t n);


//---------------------------------------------------------------------------
// Project
//---------------------------------------------------------------------------


//! Project the simplex to a lower dimension.
template<typename _T>
void
projectToLowerDimension(const std::array < std::array<_T, 2>, 1 + 1 > & s,
                        std::array < std::array<_T, 1>, 1 + 1 > * t);


//! Project the simplex to a lower dimension.
template<typename _T>
void
projectToLowerDimension(const std::array < std::array<_T, 3>, 1 + 1 > & s,
                        std::array < std::array<_T, 1>, 1 + 1 > * t);


//! Project the simplex to a lower dimension.
template<typename _T>
void
projectToLowerDimension(const std::array < std::array<_T, 3>, 2 + 1 > & s,
                        std::array < std::array<_T, 2>, 2 + 1 > * t);

//@}

} // namespace geom
}

#define __geom_mesh_simplex_geometry_ipp__
#include "stlib/geom/mesh/simplex/geometry.ipp"
#undef __geom_mesh_simplex_geometry_ipp__

#endif

// -*- C++ -*-

/*!
  \file geom/mesh/iss/fit.h
  \brief Fit a mesh to a level-set description.
*/

#if !defined(__geom_mesh_iss_fit_h__)
#define __geom_mesh_iss_fit_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/build.h"

#include <set>

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_fit Fit a mesh to a level-set description.
*/
//@{


//! Fit a mesh to a level-set description.
/*!
  \relates IndSimpSetIncAdj
*/
template<typename T, class ISS>
void
fit(IndSimpSetIncAdj<2, 1, T>* mesh,
    const ISS_SignedDistance<ISS, 2>& signedDistance,
    T deviationTangent, std::size_t numSweeps);


//! Fit the boundary of a mesh to a level-set description.
/*!
  \relates IndSimpSet
*/
template<typename T, class ISS>
void
fit(IndSimpSetIncAdj<2, 2, T>* mesh,
    const ISS_SignedDistance<ISS, 2>& signedDistance,
    T deviationTangent, std::size_t numSweeps);


//@}

} // namespace geom
}

#define __geom_mesh_iss_fit_ipp__
#include "stlib/geom/mesh/iss/fit.ipp"
#undef __geom_mesh_iss_fit_ipp__

#endif

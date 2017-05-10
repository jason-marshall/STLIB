// -*- C++ -*-

/*!
  \file procrustes.h
  \brief Ordinary Procrustes analysis.
*/

#if !defined(__geom_shape_procrustes_h__)
#define __geom_shape_procrustes_h__

#include "stlib/container/EquilateralArray.h"
#include "stlib/ext/array.h"

#include "Eigen/LU"
#include "Eigen/SVD"

#include <vector>

namespace stlib
{
namespace geom
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Use ordinary Procrustes analysis to determine the transformation that best maps one set of landmarks to another.
/*!
  \param source Input/Output. The set of source landmark points.
  \param target Input/Output. The set of target landmark points.
  \param sourceCentroid Output. The centroid of the source points.
  \param targetCentroid Output. The centroid of the target points.
  \param rotation The _D x _D rotation matrix.
  \param scale The scale factor.

  \par Template parameters.
  \c _T is the number type.
  \c _D is the space dimension for the landmark points.

  \par
  The number of landmark points in the source and target sets must be the same.
  The number of landmark points must be greater than or equal to the
  space dimension.

  \par
  This function implements the algorithm for ordinary Procrustes analysis
  presented in chapter 5 of &quot;Statistical Shape Analysis&quot;
  by I. L. Dryden and Kanti V. Mardia.

  \par
  The source and target points are centered by subtracting their centroids.
  After this centering, the source points are best mapped to the target with
  the transformation: scale * source * rotation.
*/
template<typename _T, std::size_t _D>
void
procrustes(std::vector<std::array<_T, _D> >* source,
           std::vector<std::array<_T, _D> >* target,
           std::array<_T, _D>* sourceCentroid,
           std::array<_T, _D>* targetCentroid,
           container::EquilateralArray<_T, 2, _D>* rotation,
           _T* scale);


} // namespace geom
}

#define __geom_shape_procrustes_ipp__
#include "stlib/geom/shape/procrustes.ipp"
#undef __geom_shape_procrustes_ipp__

#endif

// -*- C++ -*-

/*!
  \file decomposition.h
  \brief Functions for decomposing the Jacobian.
*/

#if !defined(__geom_decomposition_h__)
#define __geom_decomposition_h__

#include "stlib/ads/tensor/SquareMatrix.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup simplex_decompose Decompose the Jacobian
*/
//@{

//! Decompose the jacobian into orientation * skew * aspectRatio.
template<typename T>
void
decompose(const ads::SquareMatrix<2, T>& jacobian,
          ads::SquareMatrix<2, T>* orientation,
          ads::SquareMatrix<2, T>* skew,
          ads::SquareMatrix<2, T>* aspectRatio);


//! Decompose the jacobian into orientation * skew * aspectRatio.
template<typename T>
void
decompose(const ads::SquareMatrix<3, T>& jacobian,
          ads::SquareMatrix<3, T>* orientation,
          ads::SquareMatrix<3, T>* skew,
          ads::SquareMatrix<3, T>* aspectRatio);

//@}

} // namespace geom
}

#define __geom_decomposition_ipp__
#include "stlib/geom/mesh/simplex/decomposition.ipp"
#undef __geom_decomposition_ipp__

#endif

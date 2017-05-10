// -*- C++ -*-

/*!
  \file DistanceAdjDiag1st.h
  \brief Distance equation.  First-order, adjacent-diagonal scheme.
*/

#if !defined(__DistanceAdjDiag1st_h__)
#define __DistanceAdjDiag1st_h__

#include "stlib/hj/Distance.h"
#include "stlib/hj/DistanceScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Distance equation.  Adjacent difference scheme.  1st order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class DistanceAdjDiag1st;

} // namespace hj
}

#define __hj_DistanceAdjDiag1st2_h__
#include "stlib/hj/DistanceAdjDiag1st2.h"
#undef __hj_DistanceAdjDiag1st2_h__

#define __hj_DistanceAdjDiag1st3_h__
#include "stlib/hj/DistanceAdjDiag1st3.h"
#undef __hj_DistanceAdjDiag1st3_h__

#endif

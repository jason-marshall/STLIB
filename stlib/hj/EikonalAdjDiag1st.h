// -*- C++ -*-

/*!
  \file EikonalAdjDiag1st.h
  \brief Eikonal equation.  First-order, adjacent-diagonal scheme.
*/

#if !defined(__EikonalAdjDiag1st_h__)
#define __EikonalAdjDiag1st_h__

#include "stlib/hj/Eikonal.h"
#include "stlib/hj/EikonalScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Eikonal equation.  Adjacent difference scheme.  1st order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class EikonalAdjDiag1st;

} // namespace hj
}

#define __hj_EikonalAdjDiag1st2_h__
#include "stlib/hj/EikonalAdjDiag1st2.h"
#undef __hj_EikonalAdjDiag1st2_h__

#define __hj_EikonalAdjDiag1st3_h__
#include "stlib/hj/EikonalAdjDiag1st3.h"
#undef __hj_EikonalAdjDiag1st3_h__

#endif

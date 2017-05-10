// -*- C++ -*-

/*!
  \file DistanceAdjDiag2nd.h
  \brief Distance equation.  Second-order, adjacent-diagonal scheme.
*/

#if !defined(__DistanceAdjDiag2nd_h__)
#define __DistanceAdjDiag2nd_h__

#include "stlib/hj/Distance.h"
#include "stlib/hj/DistanceScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Distance equation.  Adjacent-diagonal difference scheme.  2nd order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class DistanceAdjDiag2nd;

} // namespace hj
}

#define __hj_DistanceAdjDiag2nd2_h__
#include "stlib/hj/DistanceAdjDiag2nd2.h"
#undef __hj_DistanceAdjDiag2nd2_h__

//#define __hj_DistanceAdjDiag2nd3_h__
//#include "stlib/hj/DistanceAdjDiag2nd3.h"
//#undef __hj_DistanceAdjDiag2nd3_h__

#endif

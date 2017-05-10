// -*- C++ -*-

/*!
  \file DistanceAdj2nd.h
  \brief Distance equation.  Second-order, adjacent scheme.
*/

#if !defined(__hj_DistanceAdj2nd_h__)
#define __hj_DistanceAdj2nd_h__

#include "stlib/hj/Distance.h"
#include "stlib/hj/DistanceScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Distance equation.  Adjacent difference scheme.  2nd order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class DistanceAdj2nd;

} // namespace hj
}

#define __hj_DistanceAdj2nd2_h__
#include "stlib/hj/DistanceAdj2nd2.h"
#undef __hj_DistanceAdj2nd2_h__

//#define __hj_DistanceAdj2nd3_h__
//#include "stlib/hj/DistanceAdj2nd3.h"
//#undef __hj_DistanceAdj2nd3_h__

#endif

// -*- C++ -*-

/*!
  \file EikonalAdjDiag2nd.h
  \brief Eikonal equation.  Second-order, adjacent-diagonal scheme.
*/

#if !defined(__EikonalAdjDiag2nd_h__)
#define __EikonalAdjDiag2nd_h__

#include "stlib/hj/Eikonal.h"
#include "stlib/hj/EikonalScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Eikonal equation.  Adjacent-diagonal difference scheme.  2nd order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class EikonalAdjDiag2nd;

} // namespace hj
}

#define __hj_EikonalAdjDiag2nd2_h__
#include "stlib/hj/EikonalAdjDiag2nd2.h"
#undef __hj_EikonalAdjDiag2nd2_h__

//#define __hj_EikonalAdjDiag2nd3_h__
//#include "stlib/hj/EikonalAdjDiag2nd3.h"
//#undef __hj_EikonalAdjDiag2nd3_h__

#endif

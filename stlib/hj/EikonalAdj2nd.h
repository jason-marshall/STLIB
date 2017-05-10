// -*- C++ -*-

/*!
  \file EikonalAdj2nd.h
  \brief Eikonal equation.  Second-order, adjacent scheme.
*/

#if !defined(__hj_EikonalAdj2nd_h__)
#define __hj_EikonalAdj2nd_h__

#include "stlib/hj/Eikonal.h"
#include "stlib/hj/EikonalScheme.h"

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "stlib/hj/debug.h"
#endif

namespace stlib
{
namespace hj {

//! Eikonal equation.  Adjacent difference scheme.  2nd order.
/*!
  \param N is the space dimension.
  \param T is the number type.
*/
template<std::size_t N, typename T>
class EikonalAdj2nd;

} // namespace hj
}

#define __hj_EikonalAdj2nd2_h__
#include "stlib/hj/EikonalAdj2nd2.h"
#undef __hj_EikonalAdj2nd2_h__

//#define __hj_EikonalAdj2nd3_h__
//#include "stlib/hj/EikonalAdj2nd3.h"
//#undef __hj_EikonalAdj2nd3_h__

#endif

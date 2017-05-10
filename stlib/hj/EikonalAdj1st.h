// -*- C++ -*-

/*!
  \file EikonalAdj1st.h
  \brief Eikonal equation.  First-order, adjacent scheme.
*/

#if !defined(__hj_EikonalAdj1st_h__)
#define __hj_EikonalAdj1st_h__

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
class EikonalAdj1st;

} // namespace hj
}

#define __hj_EikonalAdj1st2_h__
#include "stlib/hj/EikonalAdj1st2.h"
#undef __hj_EikonalAdj1st2_h__

#define __hj_EikonalAdj1st3_h__
#include "stlib/hj/EikonalAdj1st3.h"
#undef __hj_EikonalAdj1st3_h__

#endif

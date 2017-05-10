// -*- C++ -*-

/*!
  \file Eikonal.h
  \brief Finite difference operations for the eikonal equation in N-D.
*/

#if !defined(__hj_Eikonal_h__)
#define __hj_Eikonal_h__

#include <limits>

#include <cmath>

namespace stlib
{
namespace hj {

//! Finite differences for the eikonal equation \f$ | \nabla u | f = 1 \f$.
/*!
  \param N is the space dimension.
  \param T is the number type.

  This class does not know anything about the solution or status
  grids.  This class defines protected member functions that perform
  finite difference operations.  It provides functionality for both
  adjacent and adjacent-diagonal difference schemes.  Classes that
  derive from \c Eikonal call these low-level functions.
*/
template<std::size_t N, typename T>
class Eikonal;

} // namespace hj
}

#define __hj_Eikonal2_h__
#include "stlib/hj/Eikonal2.h"
#undef __hj_Eikonal2_h__

#define __hj_Eikonal3_h__
#include "stlib/hj/Eikonal3.h"
#undef __hj_Eikonal3_h__

#endif

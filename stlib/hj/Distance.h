// -*- C++ -*-

/*!
  \file Distance.h
  \brief Finite difference operations for computing distance in N-D.
*/

#if !defined(__hj_Distance_h__)
#define __hj_Distance_h__

#include <limits>

#include <cmath>
#include <cstddef>

namespace stlib
{
namespace hj {

//! Finite differences for computing distance.
/*!
  \param N is the space dimension.
  \param T is the number type.

  Finite differences for computing distance by solving the eikonal
  equation \f$ | \nabla u | = 1 \f$.

  This class does not know anything about the solution or status
  grids.  This class defines protected member functions that perform
  finite difference operations.  It provides functionality for both
  adjacent and adjacent-diagonal difference schemes.  Classes that
  derive from \c Distance call these low-level functions.
*/
template<std::size_t N, typename T>
class Distance;

} // namespace hj
}

#define __hj_Distance2_h__
#include "stlib/hj/Distance2.h"
#undef __hj_Distance2_h__

#define __hj_Distance3_h__
#include "stlib/hj/Distance3.h"
#undef __hj_Distance3_h__

#endif

// -*- C++ -*-

/*!
  \file DiffSchemeAdj.h
  \brief A class that supports finite difference operations for an N-D grid.

  Scheme with an adjacent stencil.
*/

#if !defined(__hj_DiffSchemeAdj_h__)
#define __hj_DiffSchemeAdj_h__

#include "stlib/hj/DiffScheme.h"

#include "stlib/ads/algorithm/min_max.h"

#include <limits>

#include <cmath>

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "debug.h"
#endif

namespace stlib
{
namespace hj {

//! Adjacent difference scheme.
/*!
  \param N is the space dimension.
  \param T is the number type.
  \param Equation represents the equation to be solved.  The equation must
  supply functions that perform the finite differencing in up to N
  adjacent directions.

  This class implements the labeling operations for adjacent difference
  schemes in the \c label_neighbors() member function.
*/
template<std::size_t N, typename T, class Equation>
class DiffSchemeAdj;

} // namespace hj
}

#define __hj_DiffSchemeAdj2_h__
#include "stlib/hj/DiffSchemeAdj2.h"
#undef __hj_DiffSchemeAdj2_h__

#define __hj_DiffSchemeAdj3_h__
#include "stlib/hj/DiffSchemeAdj3.h"
#undef __hj_DiffSchemeAdj3_h__

#endif

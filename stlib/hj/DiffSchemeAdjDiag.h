// -*- C++ -*-

/*!
  \file DiffSchemeAdjDiag.h
  \brief A class that supports finite difference operations for an N-D grid.

  Scheme with an adjacent-diagonal stencil.
*/

#if !defined(__hj_DiffSchemeAdjDiag_h__)
#define __hj_DiffSchemeAdjDiag_h__

#include "stlib/hj/DiffScheme.h"

#include <limits>

#include <cmath>

#ifdef STLIB_DEBUG
// Include the debugging code.
#include "debug.h"
#endif

namespace stlib
{
namespace hj {

//! Adjacent-diagonal difference scheme.
/*!
  \param N is the space dimension.
  \param T is the number type.
  \param Equation represents the equation to be solved.  The equation must
  supply functions that perform the finite differencing in adjacent
  and diagonal directions.

  This class implements the labeling operations for adjacent-diagonal
  difference schemes in the \c label_neighbors() member function.
*/
template<std::size_t N, typename T, class Equation>
class DiffSchemeAdjDiag;

} // namespace hj
}

#define __hj_DiffSchemeAdjDiag2_h__
#include "stlib/hj/DiffSchemeAdjDiag2.h"
#undef __hj_DiffSchemeAdjDiag2_h__

#define __hj_DiffSchemeAdjDiag3_h__
#include "stlib/hj/DiffSchemeAdjDiag3.h"
#undef __hj_DiffSchemeAdjDiag3_h__

#endif

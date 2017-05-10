/* -*- C++ -*- */

#if !defined(__levelSet_outside_h__)
#define __levelSet_outside_h__

#include "stlib/levelSet/Grid.h"

#include "stlib/numerical/integer/bits.h"

#include <cstring>

namespace stlib
{
namespace levelSet
{

//! Mark the outside grid points as negative infinity.
/*!
  The input grid is a level set function. Negative/positive distance denote
  the interior/exterior of the object. Points on the boundary with positive
  distance are defined to be outside. Any grid point with positive distance
  that is a neighbor of an outside point is also outside. This criterion
  allows us to distinguish between domains that are outside the object and
  domains that are cavities.
*/
template<typename _T>
void
markOutsideAsNegativeInf(Grid<_T, 3, 8>* grid);

} // namespace levelSet
}

#define __levelSet_outside_ipp__
#include "stlib/levelSet/outside.ipp"
#undef __levelSet_outside_ipp__

#endif

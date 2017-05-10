// -*- C++ -*-

#if !defined(__levelSet_components_h__)
#define __levelSet_components_h__

#include "stlib/levelSet/Grid.h"

#include <deque>

namespace stlib
{
namespace levelSet
{

//! Identify the connected components of the level set function.
/*!
  The grid points that are not part of a component are set to
  \c std::numeric_limits<_I>::max().
*/
template<typename _T, std::size_t _D, std::size_t N, typename _R, typename _I>
_I
labelComponents(const Grid<_T, _D, N, _R>& f,
                Grid<_I, _D, N, _R>* components);

} // namespace levelSet
}

#define __levelSet_components_ipp__
#include "stlib/levelSet/components.ipp"
#undef __levelSet_components_ipp__

#endif

// -*- C++ -*-

#if !defined(__sfc_CellFunctors_h__)
#define __sfc_CellFunctors_h__

/*!
  \file
  \brief Cell functors.
*/

#include <cstddef>
#include <cassert>

namespace stlib
{
namespace sfc
{

//! Return true if the level exceeds a threshold.
struct LevelGreaterThan {
  //! The threshold.
  std::size_t threshold;

  //! Return true if the level exceeds the threshold.
  template<typename _InputIterator>
  bool
  operator()(const std::size_t level, _InputIterator /*begin*/,
             _InputIterator /*end*/) const
  {
    return level > threshold;
  }
};


} // namespace sfc
}

#endif

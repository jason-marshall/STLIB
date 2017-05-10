// -*- C++ -*-

/*!
  \file amr/MessageStreamExtents.h
  \brief Message stream extents.
*/

#if !defined(__amr_MessageStreamExtents_h__)
#define __amr_MessageStreamExtents_h__

#include <cstddef>
#include <cassert>

namespace stlib
{
namespace amr
{

//! Return the padding required for the specified number of bytes. If the data is followed by the specified type.
template < typename _Type, typename _NextType = double >
class MessageStreamPadding
{
public:

  //! Get the padding amount.
  static
  std::size_t
  get(const std::size_t n)
  {
    return (sizeof(_NextType) - (n * sizeof(_Type) % sizeof(_NextType))) %
           sizeof(_NextType);
  }
};

//! Message stream extents.
/*!
*/
template<typename _T>
class MessageStreamExtents
{
public:

  //! The extent for a single element.
  enum {Extent = ((sizeof(_T) - 1) / sizeof(double) + 1) * sizeof(double)};

  //! The extent for an array of elements.
  static
  std::size_t
  getExtent(const std::size_t n)
  {
    return ((n * sizeof(_T) - 1) / sizeof(double) + 1) * sizeof(double);
  }
};

} // namespace amr
}

#endif

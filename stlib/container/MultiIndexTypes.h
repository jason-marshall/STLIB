// -*- C++ -*-

/*!
  \file stlib/container/MultiIndexTypes.h
  \brief Types for multi-array indexing.
*/

#if !defined(__container_MultiIndexTypes_h__)
#define __container_MultiIndexTypes_h__

#include "stlib/container/IndexTypes.h"
#include "stlib/container/MultiArrayStorage.h"

namespace stlib
{
namespace container
{

//! Types for %array indexing.
template<std::size_t _Dimension>
class MultiIndexTypes
{
  //
  // Constants.
  //
public:

  //! The number of dimensions.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;

  //
  // Public types.
  //
public:

  // Types for STL compliance.

  //! The size type.
  typedef IndexTypes::size_type size_type;
  //! Pointer difference type.
  typedef IndexTypes::difference_type difference_type;

  // Other types.

  //! An %array index is a signed integer.
  typedef IndexTypes::Index Index;
  //! A list of sizes.
  typedef std::array<size_type, _Dimension> SizeList;
  //! A list of indices.
  typedef std::array<Index, _Dimension> IndexList;
  //! The storage order.
  typedef MultiArrayStorage<_Dimension> Storage;
};

} // namespace container
}

#endif

// -*- C++ -*-

/*!
  \file stlib/container/SimpleMultiIndexRange.h
  \brief An index range.
*/

#if !defined(__container_SimpleMultiIndexRange_h__)
#define __container_SimpleMultiIndexRange_h__

#include "stlib/ext/array.h"

#include <boost/config.hpp>

namespace stlib
{
namespace container
{

USING_STLIB_EXT_ARRAY;

//! An index range.
template<std::size_t _Dimension>
struct SimpleMultiIndexRange {
  //
  // Constants.
  //

  //! The number of dimensions.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;

  //
  // Types.
  //

  //! An array index type is \c std::size_t.
  typedef std::size_t Index;
  //! A list of indices.
  typedef std::array<Index, Dimension> IndexList;

  //
  // Member data.
  //

  //! The %array extents.
  IndexList extents;
  //! The lower bound for each index.
  IndexList bases;
};


//---------------------------------------------------------------------------
// Free functions.

//! Return the intersection of the two ranges.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
SimpleMultiIndexRange<_Dimension>
overlap(const SimpleMultiIndexRange<_Dimension>& x,
        const SimpleMultiIndexRange<_Dimension>& y);

//! Return true if the index is in the index range.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
bool
isIn(const SimpleMultiIndexRange<_Dimension>& range,
     const typename SimpleMultiIndexRange<_Dimension>::IndexList& index);

//---------------------------------------------------------------------------
// Equality.

//! Return true if the member data are equal.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
inline
bool
operator==(const SimpleMultiIndexRange<_Dimension>& x,
           const SimpleMultiIndexRange<_Dimension>& y)
{
  return x.extents == y.extents && x.bases == y.bases;
}

//! Return true if they are not equal.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
inline
bool
operator!=(const SimpleMultiIndexRange<_Dimension>& x,
           const SimpleMultiIndexRange<_Dimension>& y)
{
  return !(x == y);
}

//---------------------------------------------------------------------------
// File I/O.

//! Print the extents and bases.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out, const SimpleMultiIndexRange<_Dimension>& x)
{
  out << x.extents << '\n'
      << x.bases << '\n';
  return out;
}

//! Read the extents and bases.
/*! \relates SimpleMultiIndexRange */
template<std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, SimpleMultiIndexRange<_Dimension>& x)
{
  return in >> x.extents >> x.bases;
}

} // namespace container
}

#define __container_SimpleMultiIndexRange_ipp__
#include "stlib/container/SimpleMultiIndexRange.ipp"
#undef __container_SimpleMultiIndexRange_ipp__

#endif

// -*- C++ -*-

/*!
  \file stlib/container/IndexRange.h
  \brief An index range.
*/

#if !defined(__container_IndexRange_h__)
#define __container_IndexRange_h__

#include "stlib/container/IndexTypes.h"

#include <algorithm>
#include <iostream>

#include <cassert>

namespace stlib
{
namespace container
{

//! An index range.
class
  IndexRange
{
  //
  // Types.
  //
private:

  typedef IndexTypes Types;

public:

  // Types for STL compliance.

  //! The size type.
  typedef Types::size_type size_type;

  // Other types.

  //! An array index is a signed integer.
  typedef Types::Index Index;

  //
  // Member data.
  //
protected:

  //! The array extents.
  size_type _extent;
  //! The lower bound for the index.
  Index _base;
  //! The step.
  Index _step;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Default value data.
  IndexRange() :
    _extent(),
    _base(),
    _step()
  {
  }

  //! Construct from the extent, and optionally the base and the step.
  IndexRange(const size_type extent, const Index base = 0,
             const Index step = 1) :
    _extent(extent),
    _base(base),
    _step(step)
  {
  }

  // The default copy constructor, assignment operator, and destructor are fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The index range extent.
  size_type
  extent() const
  {
    return _extent;
  }

  //! The base.
  Index
  base() const
  {
    return _base;
  }

  //! The step.
  Index
  step() const
  {
    return _step;
  }

  //@}
};


//---------------------------------------------------------------------------
// Free functions.

//! Return the intersection of the two index ranges.
/*! \relates IndexRange */
IndexRange
overlap(const IndexRange::size_type extent1,
        const IndexRange::Index base1,
        const IndexRange::size_type extent2,
        const IndexRange::Index base2);

//! Return the intersection of the two ranges.
/*!
  \pre The ranges must have unit steps.
  \relates IndexRange
*/
IndexRange
overlap(const IndexRange& x, const IndexRange& y);

//! Return true if the index is in the index range.
/*! \relates IndexRange */
bool
isIn(const IndexRange& range, const IndexRange::Index index);

//---------------------------------------------------------------------------
// Equality.

//! Return true if the member data are equal.
/*! \relates IndexRange */
inline
bool
operator==(const IndexRange& x, const IndexRange& y)
{
  return x.extent() == y.extent() && x.base() == y.base() &&
         x.step() == y.step();
}

//! Return true if they are not equal.
/*! \relates IndexRange */
inline
bool
operator!=(const IndexRange& x, const IndexRange& y)
{
  return !(x == y);
}

//---------------------------------------------------------------------------
// File I/O.

//! Print the extents, bases, and steps.
/*! \relates IndexRange */
inline
std::ostream&
operator<<(std::ostream& out, const IndexRange& x)
{
  out << x.extent() << '\n'
      << x.base() << '\n'
      << x.step() << '\n';
  return out;
}

//! Read the extents, bases, and steps.
/*! \relates IndexRange */
inline
std::istream&
operator>>(std::istream& in, IndexRange& x)
{
  IndexRange::size_type extent;
  IndexRange::Index base, step;
  in >> extent >> base >> step;
  x = IndexRange(extent, base, step);
  return in;
}

} // namespace container
}

#define __container_IndexRange_ipp__
#include "stlib/container/IndexRange.ipp"
#undef __container_IndexRange_ipp__

#endif

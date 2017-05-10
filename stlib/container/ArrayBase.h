// -*- C++ -*-

/*!
  \file stlib/container/ArrayBase.h
  \brief Base class for arrays.
*/

#if !defined(__container_ArrayBase_h__)
#define __container_ArrayBase_h__

#include "stlib/container/IndexTypes.h"
#include "stlib/container/IndexRange.h"

namespace stlib
{
namespace container
{

//! Base class for arrays.
class
  ArrayBase
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
  //! Pointer difference type.
  typedef Types::difference_type difference_type;

  // Other types.

  //! An array index is a signed integer.
  typedef Types::Index Index;
  //! An index range.
  typedef IndexRange Range;

  //
  // Member data.
  //
protected:

  //! The number of elements.
  size_type _size;
  //! The lower bound.
  Index _base;
  //! The stride for indexing.
  Index _stride;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor and assignment operator are fine.

  //! Construct from the array size, the index bases, the storage order, and the strides.
  ArrayBase(const size_type size, const Index base, const Index stride) :
    _size(size),
    _base(base),
    _stride(stride)
  {
  }

  //! Destructor does nothing.
  virtual
  ~ArrayBase()
  {
  }

protected:

  //! Rebuild the data structure.
  void
  rebuild(const size_type size, const Index base, const Index stride)
  {
    _size = size;
    _base = base;
    _stride = stride;
  }

private:

  //! Default constructor not implemented.
  ArrayBase();

  //@}
  //--------------------------------------------------------------------------
  //! \name Random Access Container.
  //@{
public:

  //! Return true if the range is empty.
  bool
  empty() const
  {
    return _size == 0;
  }

  //! Return the size (number of elements) of the range.
  size_type
  size() const
  {
    return _size;
  }

  //! Return the size of the range.
  /*! The the max_size and the size are the same. */
  size_type
  max_size() const
  {
    return size();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  //! The index lower bound.
  Index
  base() const
  {
    return _base;
  }

  //! Set the index lower bound.
  void
  setBase(const Index base)
  {
    _base = base;
  }

  //! The index range.
  Range
  range() const
  {
    return Range(size(), base());
  }

  //! The stride for indexing.
  Index
  stride() const
  {
    return _stride;
  }

  //! The offset for indexing the base.
  difference_type
  offset() const
  {
    return _base * _stride;
  }

protected:

  //! Return the array index for the given index list.
  /*!
    For arrays with contiguous storage, this index is in the range
    [0..size()-1].
  */
  Index
  arrayIndex(const Index index) const
  {
    return (index - _base) * _stride;
  }

  //@}
};

// CONTINUE: Doxygen thinks these equality operators clash with those
// defined in other groups. I don't know why.
//----------------------------------------------------------------------------
//! \defgroup ArrayBaseEquality Equality Operators
//@{

//! Return true if the member data are equal.
/*! \relates ArrayBase */
inline
bool
operator==(const ArrayBase& x, const ArrayBase& y)
{
  return x.size() == y.size() && x.base() == y.base() &&
         x.stride() == y.stride();
}

//! Return true if they are not equal.
/*! \relates ArrayBase */
inline
bool
operator!=(const ArrayBase& x, const ArrayBase& y)
{
  return !(x == y);
}

//@}

} // namespace container
}

#endif

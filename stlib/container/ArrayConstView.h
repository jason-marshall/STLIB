// -*- C++ -*-

/*!
  \file stlib/container/ArrayConstView.h
  \brief Constant view of an array.
*/

#if !defined(__container_ArrayConstView_h__)
#define __container_ArrayConstView_h__

#include "stlib/container/ArrayTypes.h"
#include "stlib/container/ArrayBase.h"
#include "stlib/container/ViewIterator.h"

#include <numeric>
#include <algorithm>

namespace stlib
{
namespace container
{

//! Constant view of an %array.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor. This class uses the automatically-generated
  copy constructor; the array data is referenced. You can create
  an instance of this class with the view() member function.

  The copy constructors create shallow copies of the argument, i.e. the
  array data is referenced.
  \code
  container::Array<int> a(extent);
  container::ArrayConstView<int> b(a);
  \endcode
  The argument may be any %array type, however the value type must be the same.

  <b>Container Member Functions</b>

  ArrayConstView inherits the following functionality for treating
  the %array as a random access container.

  - ArrayBase::empty()
  - ArrayBase::size()
  - ArrayBase::max_size()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()

  <b>%Array Indexing Member Functions</b>

  ArrayConstView inherits the following %array indexing functionality.

  - ArrayBase::extent()
  - ArrayBase::base()
  - ArrayBase::setBase()
  - ArrayBase::range()
  - ArrayBase::stride()
  - ArrayBase::offset()

  It defines the following functions.

  - operator()()
  - data()
  - view()

  <b>Free Functions</b>

*/
template<typename _T>
class
  ArrayConstView : public ArrayBase
{
  //
  // Types.
  //
private:

  typedef ArrayBase Base;
  typedef ArrayTypes<_T> Types;

public:

  // Types for STL compliance.

  //! The element type of the array.
  typedef typename Types::value_type value_type;
  //! A pointer to a constant array element.
  typedef typename Types::const_pointer const_pointer;
  //! An iterator on constant elements in the array.
  typedef ViewIterator<ArrayConstView, true> const_iterator;
  //! A reverse iterator on constant elements in the array.
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  //! A reference to a constant array element.
  typedef typename Types::const_reference const_reference;
  //! The size type.
  typedef typename Types::size_type size_type;
  //! Pointer difference type.
  typedef typename Types::difference_type difference_type;

  // Other types.

  //! The parameter type.
  /*! This is used for passing the value type as an argument. */
  typedef typename Types::Parameter Parameter;
  //! An array index is a signed integer.
  typedef typename Types::Index Index;
  //! An index range.
  typedef typename Base::Range Range;
  //! A constant view of this array.
  typedef ArrayConstView<_T> ConstView;

  //
  // Member data.
  //
protected:

  //! Pointer to the beginning of a contiguous block of data.
  const_pointer _constData;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor is fine.

  //! Construct from a pointer to the memory, the array size, the index base, and the stride.
  ArrayConstView(const_pointer data, const size_type size, const Index base,
                 const Index stride) :
    Base(size, base, stride),
    _constData(data)
  {
  }

  //! Destructor does not deallocate memory.
  virtual
  ~ArrayConstView()
  {
  }

protected:

  //! Rebuild the data structure.
  void
  rebuild(const_pointer data, const size_type size, const Index base,
          const Index stride)
  {
    _constData = data;
    Base::rebuild(size, base, stride);
  }

  //! Copy the data structure. Shallow copy of the elements.
  void
  rebuild(const ArrayConstView& x)
  {
    rebuild(x.data(), x.size(), x.base(), x.stride());
  }

private:

  //! Default constructor not implemented.
  /*!
    This class is a virtual base for other classes. Making the default
    constructor private makes sure this class is appropriately constructed.
  */
  ArrayConstView();

  //! Assignment operator not implemented. You cannot assign to const data.
  ArrayConstView&
  operator=(const ArrayConstView& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Random Access Container.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;

  //! Return a const iterator to the first value.
  const_iterator
  begin() const
  {
    return const_iterator::begin(*this);
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return const_iterator::end(*this);
  }

  //! Return a const reverse iterator to the end of the sequence.
  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  //! Return a const reverse iterator to the beginning of the sequence.
  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  using Base::base;
  using Base::setBase;
  using Base::range;
  using Base::stride;
  using Base::offset;

  //! Array indexing.
  const_reference
  operator()(const Index index) const
  {
    return _constData[arrayIndex(index)];
  }

  //! Return a const pointer to the beginning of the data.
  const_pointer
  data() const
  {
    return _constData;
  }

  //! Make a sub-array view with the specified index range.
  /*! The bases for the view are the same as that for the index range. */
  ConstView
  view(const Range& range) const
  {
    return ConstView(&(*this)(range.base()), range.extent(), range.base(),
                     stride() * range.step());
  }

protected:

  using Base::arrayIndex;

  //! Set the data pointer.
  void
  setData(const_pointer data)
  {
    _constData = data;
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup arrayArrayConstViewMathematical ArrayConstView Mathematical Functions
//@{

//! Return the sum of the components.
/*! \relates ArrayConstView */
template<typename _T>
inline
_T
sum(const ArrayConstView<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

//! Return the product of the components.
/*! \relates ArrayConstView */
template<typename _T>
inline
_T
product(const ArrayConstView<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
/*! \relates ArrayConstView */
template<typename _T>
inline
_T
min(const ArrayConstView<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

//! Return the maximum component.  Use > for comparison.
/*! \relates ArrayConstView */
template<typename _T>
inline
_T
max(const ArrayConstView<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

//@}

// CONTINUE: Add equality and file output.

} // namespace container
}

#endif

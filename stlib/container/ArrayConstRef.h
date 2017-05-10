// -*- C++ -*-

/*!
  \file stlib/container/ArrayConstRef.h
  \brief Constant array that references memory and has contiguous storage.
*/

#if !defined(__container_ArrayConstRef_h__)
#define __container_ArrayConstRef_h__

#include "stlib/container/ArrayConstView.h"

namespace stlib
{
namespace container
{

//! Constant %array that references memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor.

  You can construct an %array from a const pointer to the data and its
  size.
  Below we make an %array with index range [0..7]
  \code
  double data[8];
  ...
  container::ArrayConstRef<double> a(data, sizeof(data) / sizeof(double));
  \endcode

  You can also specify the index bases. Below we make an
  %array with index range [-3..4]
  \code
  double data[8];
  ...
  container::ArrayConstRef<double, 3> a(data, 8, -3);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  array data is referenced.
  \code
  container::Array<int> a(extents);
  container::ArrayConstRef<int> b(a);
  \endcode
  The argument may be a Array, ArrayRef, or a ArrayConstRef.
  The dimension and value type must be the same.

  Since this is a constant %array class, there are no assignment operators.

  You can use rebuild() to make a constant reference to another %array.
  \code
  container::Array<int> a(size);
  container::ArrayConstRef<int> b(a);
  container::Array<int> c(size);
  b.rebuild(c);
  \endcode

  <b>Container Member Functions</b>

  ArrayConstRef inherits the following functionality for treating the
  %array as a constant random access container.

  - ArrayBase::empty()
  - ArrayBase::size()
  - ArrayBase::max_size()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()

  <b>%Array Indexing Member Functions</b>

  ArrayConstRef inherits the following %array indexing functionality.

  - ArrayBase::base()
  - ArrayBase::setBase()
  - ArrayBase::range()
  - ArrayBase::stride()
  - ArrayBase::offset()
  - ArrayView::operator()()
  - ArrayView::view()

  <b>Free Functions</b>

  - \ref ArrayConstRefEquality
  - \ref ArrayConstRefFile
*/
template<typename _T>
class
  ArrayConstRef :
  virtual public ArrayConstView<_T>
{
  //
  // Types.
  //
private:

  typedef ArrayConstView<_T> Base;
  typedef ArrayTypes<_T> Types;

public:

  // Types for STL compliance.

  //! The element type of the array.
  typedef typename Types::value_type value_type;
  //! A pointer to a constant array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on constant elements in the array.
  typedef typename Types::const_iterator const_iterator;
  //! A reverse iterator on constant elements in the array.
  typedef typename Types::const_reverse_iterator const_reverse_iterator;
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
  typedef typename Base::ConstView ConstView;

  //
  // Using member data.
  //
protected:

  using Base::_constData;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor is fine.

  //! Construct from a pointer to the memory and the array extents.
  ArrayConstRef(const_pointer data, const size_type size) :
    Base(data, size, 0, 1)
  {
  }

  //! Construct from a pointer to the memory, the array extents, the index bases, and optionally the storage order.
  ArrayConstRef(const_pointer data, const size_type size, const Index base) :
    Base(data, size, base, 1)
  {
  }

  //! Destructor does not deallocate memory.
  virtual
  ~ArrayConstRef()
  {
  }

  //! Copy the data structure. Shallow copy of the elements.
  void
  rebuild(const ArrayConstRef& x)
  {
    Base::rebuild(x.data(), x.size(), x.base(), x.stride());
  }

  //! Rebuild the data structure.
  void
  rebuild(const Index base)
  {
    rebuild(_constData, size(), base);
  }

  //! Rebuild the data structure.
  void
  rebuild(const_pointer data, const size_type size, const Index base)
  {
    Base::rebuild(data, size, base, 1);
  }

private:

  //! Default constructor not implemented.
  ArrayConstRef()
  {
  }

  //! Assignment operator not implemented. You cannot assign to const data.
  ArrayConstRef&
  operator=(const ArrayConstRef& other);

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
    return data();
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return data() + size();
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

  //! Container indexing.
  const_reference
  operator[](const size_type n) const
  {
    return data()[n];
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
  using Base::data;
  using Base::view;

protected:

  using Base::arrayIndex;
  using Base::setData;

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup ArrayConstRefEquality Equality and Comparison Operators
//@{

//! Return true if the arrays have the same extents and elements.
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator==(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return x.size() == y.size() &&
         std::equal(x.begin(), x.end(), y.begin());
}

//! Return true if they are not equal.
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator!=(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return !(x == y);
}


//! Lexicographical comparison of the elements.
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator<(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

//! Return y < x.
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator>(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator<=(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates ArrayConstRef */
template<typename _T>
inline
bool
operator>=(const ArrayConstRef<_T>& x, const ArrayConstRef<_T>& y)
{
  return !(x < y);
}

//@}
//----------------------------------------------------------------------------
//! \defgroup ArrayConstRefFile File I/O
//@{

//! Print the size, base, and elements.
/*! \relates ArrayConstRef */
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const ArrayConstRef<_T>& x)
{
  out << x.size() << '\n'
      << x.base() << '\n';
  std::copy(x.begin(), x.end(), std::ostream_iterator<_T>(out, "\n"));
  return out;
}

//@}
//----------------------------------------------------------------------------
/*! \defgroup arrayArrayConstRefMathematical ArrayConstRef Mathematical Functions

  Note that these functions are defined for
  \ref arrayArrayConstRefMathematical "ArrayConstView". We redefine
  them for ArrayConstRef because this class has more efficient
  iterators.
*/
//@{

//! Return the sum of the components.
/*! \relates ArrayConstRef */
template<typename _T>
inline
_T
sum(const ArrayConstRef<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

//! Return the product of the components.
/*! \relates ArrayConstRef */
template<typename _T>
inline
_T
product(const ArrayConstRef<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
/*! \relates ArrayConstRef */
template<typename _T>
inline
_T
min(const ArrayConstRef<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

//! Return the maximum component.  Use > for comparison.
/*! \relates ArrayConstRef */
template<typename _T>
inline
_T
max(const ArrayConstRef<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

//@}

} // namespace container
}

#endif

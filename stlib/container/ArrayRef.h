// -*- C++ -*-

/*!
  \file stlib/container/ArrayRef.h
  \brief %Array that references memory and has contiguous storage.
*/

#if !defined(__container_ArrayRef_h__)
#define __container_ArrayRef_h__

#include "stlib/container/ArrayConstRef.h"
#include "stlib/container/ArrayView.h"

namespace stlib
{
namespace container
{

//! %Array that references memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor.

  You can construct an %array from a pointer to the data and its size.
  Below we make an %array with index range [0..7]
  \code
  double data[8];
  container::ArrayRef<double> a(data, 8);
  \endcode

  You can also specify the index base. Below we make an
  %array with index range [-3..4]
  \code
  double data[8];
  container::ArrayRef<double> a(data, 8, -3);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  array data is referenced.
  \code
  container::Array<int> a(size);
  container::ArrayRef<int> b(a);
  \endcode
  The argument may be a Array, or a ArrayRef.
  The value type must be the same.

  The assignment operators copy the element values. The argument must have
  the same index ranges as the %array, though they can differ in the value
  type.
  \code
  container::Array<int> a(size);
  {
    int* data = new int[size];
    container::ArrayRef<int> b(data, size);
    b = a;
  }
  {
    double* data = new double[size];
    container::Array<double> c(data, size);
    c = a;
  }
  \endcode
  The argument may be any of the multidimensional %array types.

  You can use rebuild() to make a reference to another %array.
  \code
  container::Array<int> a(size);
  container::ArrayRef<int> b(a);
  container::Array<int> c(size);
  b.rebuild(c);
  \endcode

  <b>Container Member Functions</b>

  ArrayRef inherits the following functionality for treating the %array as
  a constant random access container.

  - ArrayBase::empty()
  - ArrayBase::size()
  - ArrayBase::max_size()
  - ArrayConstRef::begin()
  - ArrayConstRef::end()
  - ArrayConstRef::rbegin()
  - ArrayConstRef::rend()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()
  - fill()

  <b>%Array Indexing Member Functions</b>

  ArrayRef inherits the following %array indexing functionality.

  - ArrayBase::base()
  - ArrayBase::setBase()
  - ArrayBase::range()
  - ArrayBase::stride()
  - ArrayBase::offset()
  - ArrayView::operator()()
  - ArrayView::view()

  It defines the following functions.

  - data()

  <b>Free Functions</b>

  - \ref ArrayRefAssignmentOperatorsScalar
  - \ref ArrayConstRefEquality
  - \ref ArrayConstRefFile
*/
template<typename _T>
class
  ArrayRef : public ArrayConstRef<_T>,
  public ArrayView<_T>
{
  //
  // Types.
  //
private:

  typedef ArrayTypes<_T> Types;
  typedef ArrayConstRef<_T> Base;
  typedef ArrayView<_T> ViewBase;
  typedef ArrayConstView<_T> VirtualBase;

public:

  // Types for STL compliance.

  //! The element type of the array.
  typedef typename Types::value_type value_type;
  //! A pointer to an array element.
  typedef typename Types::pointer pointer;
  //! A pointer to a constant array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on elements in the array.
  typedef typename Types::iterator iterator;
  //! A iterator on constant elements in the array.
  typedef typename Types::const_iterator const_iterator;
  //! A reverse iterator on elements in the array.
  typedef typename Types::reverse_iterator reverse_iterator;
  //! A reverse iterator on constant elements in the array.
  typedef typename Types::const_reverse_iterator const_reverse_iterator;
  //! A reference to an array element.
  typedef typename Types::reference reference;
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
  typedef typename ViewBase::ConstView ConstView;
  //! A view of this array.
  typedef typename ViewBase::View View;

  //
  // Using member data.
  //
protected:

  using ViewBase::_data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Copy constructor.
  ArrayRef(const ArrayRef& other);

  //! Construct from a pointer to the memory and the size.
  ArrayRef(pointer data, size_type size);

  //! Construct from a pointer to the memory, the size, and the index base.
  ArrayRef(pointer data, size_type size, Index base);

  //! Assignment operator for other array views.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  ArrayRef&
  operator=(const ArrayConstView<_T2>& other);

  //! Assignment operator for arrays with contiguous memory.
  /*!
    \pre The arrays must have the same index range.
    \note This version is faster than the assignment operator that takes a
    ArrayConstView as an argument because arrays with contiguous memory
    have faster iterators.
  */
  template<typename _T2>
  ArrayRef&
  operator=(const ArrayConstRef<_T2>& other);

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  ArrayRef&
  operator=(const ArrayRef& other);

  //! Destructor does not deallocate memory.
  virtual
  ~ArrayRef()
  {
  }

  //! Copy the data structure. Shallow copy of the elements.
  void
  rebuild(ArrayRef& x)
  {
    ViewBase::rebuild(x.data(), x.size(), x.base());
  }

  //! Rebuild the data structure.
  void
  rebuild(const Index base)
  {
    rebuild(data(), size(), base);
  }

  void
  rebuild(pointer data, size_type size, Index base)
  {
    ViewBase::rebuild(data, size, base);
  }

private:

  //! Default constructor not implemented.
  ArrayRef();

  //@}
  //--------------------------------------------------------------------------
  //! \name Random Access Container.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;
  using Base::begin;
  using Base::end;
  using Base::rbegin;
  using Base::rend;
  using Base::operator[];

  //! Return an iterator to the first value.
  iterator
  begin()
  {
    return _data;
  }

  //! Return an iterator to one past the last value.
  iterator
  end()
  {
    return _data + size();
  }

  //! Return a reverse iterator to the end of the sequence.
  reverse_iterator
  rbegin()
  {
    return reverse_iterator(end());
  }

  //! Return a reverse iterator to the beginning of the sequence.
  reverse_iterator
  rend()
  {
    return reverse_iterator(begin());
  }

  //! Container indexing.
  reference
  operator[](const size_type n)
  {
    return _data[n];
  }

  //! Fill the array with the specified value.
  template<typename _T2>
  void
  fill(const _T2& value)
  {
    std::fill(begin(), end(), value);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing accessors.
  //@{
public:

  using Base::base;
  using Base::setBase;
  using Base::range;
  using Base::stride;
  using Base::offset;
  using Base::data;
  using ViewBase::view;

  //! Return a pointer to the beginning of the data.
  pointer
  data()
  {
    return _data;
  }

protected:

  using Base::arrayIndex;
  using ViewBase::setData;

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup ArrayRefAssignmentOperatorsScalar Assignment Operators with Scalar Operand
//@{

//! Array-scalar addition.
/*! \relates ArrayRef */
template<typename _T>
inline
ArrayRef<_T>&
operator+=(ArrayRef<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i += value;
  }
  return x;
}

//! Array-scalar subtraction.
/*! \relates ArrayRef */
template<typename _T>
inline
ArrayRef<_T>&
operator-=(ArrayRef<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i -= value;
  }
  return x;
}

//! Array-scalar multiplication.
/*! \relates ArrayRef */
template<typename _T>
inline
ArrayRef<_T>&
operator*=(ArrayRef<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i *= value;
  }
  return x;
}

//! Array-scalar division.
/*!
  \relates ArrayRef
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T>
inline
ArrayRef<_T>&
operator/=(ArrayRef<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i /= value;
  }
  return x;
}

//! Array-scalar modulus.
/*!
  \relates ArrayRef
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T>
inline
ArrayRef<_T>&
operator%=(ArrayRef<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i %= value;
  }
  return x;
}

//! Left shift.
/*! \relates ArrayRef */
template<typename _T>
inline
ArrayRef<_T>&
operator<<=(ArrayRef<_T>& x, const int offset)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i <<= offset;
  }
  return x;
}

//! Right shift.
/*! \relates ArrayRef */
template<typename _T>
inline
ArrayRef<_T>&
operator>>=(ArrayRef<_T>& x, const int offset)
{
  typedef typename ArrayRef<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i >>= offset;
  }
  return x;
}

//@}

//---------------------------------------------------------------------------
// File I/O.

// CONTINUE: Add input.

} // namespace container
}

#define __container_ArrayRef_ipp__
#include "stlib/container/ArrayRef.ipp"
#undef __container_ArrayRef_ipp__

#endif

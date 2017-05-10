// -*- C++ -*-

/*!
  \file stlib/container/ArrayView.h
  \brief View of an array.
*/

#if !defined(__container_ArrayView_h__)
#define __container_ArrayView_h__

#include "stlib/container/ArrayConstView.h"

namespace stlib
{
namespace container
{

// Forward declaration for assignment operator.
template<typename _T>
class ArrayConstRef;

//! View of an %array.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor. This class uses the automatically-generated
  copy constructor; the array data is referenced. Ordinarilly one would create
  an instance of this class with the view() member function. However,
  one can also use the constructors.

  You can construct an %array from a pointer to the data and its size.
  Below we make an %array with index range [0..7]
  \code
  double data[8];
  container::ArrayConstView<double> a(data, 8);
  \endcode

  You can also specify the index bases. Below we make an
  %array with index range [-3..4]
  \code
  double data[8];
  container::ArrayConstView<double> a(data, 8, -3);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  array data is referenced.
  \code
  container::Array<int> a(size);
  container::ArrayView<int> b(a);
  \endcode
  The argument may be a Array, a ArrayRef, or a ArrayView.
  The dimension and value type must be the same.

  The assignment operators copy the element values. The argument must have
  the same index range as the %array, though they can differ in the value
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

  <b>Container Member Functions</b>

  ArrayView inherits the following functionality for treating the %array as
  a random access container.

  - ArrayBase::empty()
  - ArrayBase::size()
  - ArrayBase::max_size()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()
  - fill()

  <b>%Array Indexing Member Functions</b>

  ArrayView inherits the following %array indexing functionality.

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

  - \ref ArrayViewAssignmentOperatorsScalar
*/
template<typename _T>
class
  ArrayView :
  virtual public ArrayConstView<_T>
{
  //
  // Types.
  //
private:

  typedef ArrayTypes<_T> Types;
  typedef ArrayConstView<_T> Base;

public:

  // Types for STL compliance.

  //! The element type of the array.
  typedef typename Types::value_type value_type;
  //! A pointer to an array element.
  typedef typename Types::pointer pointer;
  //! A pointer to a constant array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on elements in the array.
  typedef ViewIterator<ArrayView, false> iterator;
  //! A reverse iterator on elements in the array.
  typedef std::reverse_iterator<iterator> reverse_iterator;
  //! A iterator on constant elements in the array.
  typedef ViewIterator<ArrayView, true> const_iterator;
  //! A reverse iterator on constant elements in the array.
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
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
  typedef typename Base::ConstView ConstView;
  //! A view of this array.
  typedef ArrayView View;

  //
  // Member data.
  //
protected:

  //! Pointer to the data.
  pointer _data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor is fine.

  //! Construct from a pointer to the memory, the array extents, the index bases, the storage order, and the strides.
  ArrayView(pointer data, const size_type size, const Index base,
            const Index stride) :
    Base(data, size, base, stride),
    _data(data)
  {
  }

  //! Assignment operator for other array views.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  ArrayView&
  operator=(const ArrayConstView<_T2>& other)
  {
#ifdef STLIB_DEBUG
    // The arrays must have the same index range.
    assert(size() == other.size() && base() == other.base());
#endif
    // Copy the elements.
    std::copy(other.begin(), other.end(), begin());
    return *this;
  }

  //! Assignment operator for arrays with contiguous memory.
  /*!
    \pre The arrays must have the same index range.
    \note This version is faster than the assignment operator that takes a
    ArrayConstView as an argument because arrays with contiguous memory
    have faster iterators.
  */
  template<typename _T2>
  ArrayView&
  operator=(const ArrayConstRef<_T2>& other)
  {
#ifdef STLIB_DEBUG
    // The arrays must have the same index range.
    assert(size() == other.size() && base() == other.base());
#endif
    // Copy the elements.
    std::copy(other.begin(), other.end(), begin());
    return *this;
  }

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  ArrayView&
  operator=(const ArrayView& other)
  {
    if (this != &other) {
#ifdef STLIB_DEBUG
      // The arrays must have the same index range.
      assert(size() == other.size() && base() == other.base());
#endif
      // Copy the elements.
      std::copy(other.begin(), other.end(), begin());
    }
    return *this;
  }

  //! Destructor does not deallocate memory.
  virtual
  ~ArrayView()
  {
  }

protected:

  //! Rebuild the data structure.
  void
  rebuild(pointer data, const size_type size, const Index base)
  {
    Base::rebuild(data, size, base, 1);
    _data = data;
  }

private:

  //! Default constructor not implemented.
  ArrayView();

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

  //! Return an iterator to the first value.
  iterator
  begin()
  {
    return iterator::begin(*this);
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return const_iterator::end(*this);
  }

  //! Return an iterator to one past the last value.
  iterator
  end()
  {
    return iterator::end(*this);
  }

  //! Return a const reverse iterator to the end of the sequence.
  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  //! Return a reverse iterator to the end of the sequence.
  reverse_iterator
  rbegin()
  {
    return reverse_iterator(end());
  }

  //! Return a const reverse iterator to the beginning of the sequence.
  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  //! Return a reverse iterator to the beginning of the sequence.
  reverse_iterator
  rend()
  {
    return reverse_iterator(begin());
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
  //! \name Array indexing.
  //@{
public:

  using Base::base;
  using Base::setBase;
  using Base::range;
  using Base::stride;
  using Base::offset;
  using Base::operator();
  using Base::data;
  using Base::view;

  //! Array indexing.
  reference
  operator()(const Index index)
  {
    return _data[arrayIndex(index)];
  }

  //! Return a pointer to the beginning of the data.
  pointer
  data()
  {
    return _data;
  }

  //! Make a sub-array view with the index range and optionally the specified bases.
  /*! The bases for the view are the same as that for the index range. */
  View
  view(const Range& range)
  {
    return View(&(*this)(range.base()), range.extent(), range.base(),
                stride() * range.step());
  }

protected:

  using Base::arrayIndex;

  //! Set the data pointer.
  void
  setData(pointer data)
  {
    Base::setData(data);
    _data = data;
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup ArrayViewAssignmentOperatorsScalar Assignment Operators with Scalar Operand
//@{

//! Array-scalar addition.
/*! \relates ArrayView */
template<typename _T>
inline
ArrayView<_T>&
operator+=(ArrayView<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i += value;
  }
  return x;
}

//! Array-scalar subtraction.
/*! \relates ArrayView */
template<typename _T>
inline
ArrayView<_T>&
operator-=(ArrayView<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i -= value;
  }
  return x;
}

//! Array-scalar multiplication.
/*! \relates ArrayView */
template<typename _T>
inline
ArrayView<_T>&
operator*=(ArrayView<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i *= value;
  }
  return x;
}

//! Array-scalar division.
/*!
  \relates ArrayView
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T>
inline
ArrayView<_T>&
operator/=(ArrayView<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i /= value;
  }
  return x;
}

//! Array-scalar modulus.
/*!
  \relates ArrayView
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T>
inline
ArrayView<_T>&
operator%=(ArrayView<_T>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i %= value;
  }
  return x;
}

//! Left shift.
/*! \relates ArrayView */
template<typename _T>
inline
ArrayView<_T>&
operator<<=(ArrayView<_T>& x, const int offset)
{
  typedef typename ArrayView<_T>::iterator iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i <<= offset;
  }
  return x;
}

//! Right shift.
/*! \relates ArrayView */
template<typename _T>
inline
ArrayView<_T>&
operator>>=(ArrayView<_T>& x, const int offset)
{
  typedef typename ArrayView<_T>::iterator iterator;
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

#endif

// -*- C++ -*-

/*!
  \file stlib/container/SimpleMultiArrayRef.h
  \brief Multi-dimensional %array that references memory and has contiguous storage.
*/

#if !defined(__container_SimpleMultiArrayRef_h__)
#define __container_SimpleMultiArrayRef_h__

#include "stlib/container/SimpleMultiArrayConstRef.h"

namespace stlib
{
namespace container
{

//! Multi-dimensional %array that references memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor.
  You can construct an %array from a pointer to the data and its index extents.
  Below we make a 2x4x8 %array with index range [0..1]x[0..3]x[0..7]
  \code
  double data[2 * 4 * 8];
  container::SimpleMultiArrayRef<double, 3>::SizeList extents(2, 4, 8)
  container::SimpleMultiArrayRef<double, 3> a(data, extents);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  %array data is referenced.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArrayRef<int, 3> b(a);
  \endcode
  The argument may be a SimpleMultiArray, or a SimpleMultiArrayRef.
  The dimension and value type must be the same.

  The assignment operators copy the element values. The argument must have
  the same index ranges as the %array, though they can differ in the value
  type.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  {
    int* data = new int[product(extents)];
    container::SimpleMultiArrayRef<int, 3> b(data, extents);
    b = a;
  }
  {
    double* data = new double[product(extents)];
    container::SimpleMultiArray<double, 3> c(data, extents);
    c = a;
  }
  \endcode
  The argument may be any of the multidimensional %array types.

  You can use rebuild() to make a reference to another %array.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArrayRef<int, 3> b(a);
  container::SimpleMultiArray<int, 3> c(extents);
  b.rebuild(c);
  \endcode

  <b>Container Member Functions</b>

  SimpleMultiArrayRef inherits the following functionality for treating
  the %array as a constant random access container.
  - SimpleMultiArrayConstRef::empty()
  - SimpleMultiArrayConstRef::size()
  - SimpleMultiArrayConstRef::max_size()
  - SimpleMultiArrayConstRef::begin()
  - SimpleMultiArrayConstRef::end()
  - SimpleMultiArrayConstRef::rbegin()
  - SimpleMultiArrayConstRef::rend()

  It defines the following functions.
  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()
  - fill()

  <b>%Array Indexing Member Functions</b>

  SimpleMultiArrayRef inherits the following %array indexing functionality.
  - SimpleMultiArrayConstRef::extents()
  - SimpleMultiArrayConstRef::strides()
  - SimpleMultiArrayConstRef::operator()()

  It defines the following functions.
  - data()

  <b>Free Functions</b>
  - \ref SimpleMultiArrayRefAssignmentOperatorsScalar
  - \ref SimpleMultiArrayConstRefEquality
  - \ref SimpleMultiArrayConstRefFile
  - \ref SimpleMultiArrayRefFile
*/
template<typename _T, std::size_t _Dimension>
class SimpleMultiArrayRef :
  public SimpleMultiArrayConstRef<_T, _Dimension>
{
  //
  // Types.
  //
private:

  typedef SimpleMultiArrayConstRef<_T, _Dimension> Base;

public:

  // Types for STL compliance.

  //! A pointer to an %array element.
  typedef typename Base::value_type* pointer;
  //! An iterator in the %array.
  typedef typename Base::value_type* iterator;
  //! A reverse iterator in the %array.
  typedef std::reverse_iterator<iterator> reverse_iterator;
  //! A reference to an %array element.
  typedef typename Base::value_type& reference;

  //
  // Member data.
  //
protected:

  //! Pointer to the data.
  pointer _data;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor.
    The default constructor is not implemented.
  */
  //@{
public:

  //! Construct from a pointer to the memory and the %array extents.
  SimpleMultiArrayRef(pointer data, const typename Base::IndexList& extents);

  //! Rebuild using a pointer to the memory and the %array extents.
  void
  rebuild(pointer data, const typename Base::IndexList& extents);

  //! Assignment operator for arrays with contiguous memory.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  SimpleMultiArrayRef&
  operator=(const SimpleMultiArrayConstRef<_T2, _Dimension>& other);

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  SimpleMultiArrayRef&
  operator=(const SimpleMultiArrayRef& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Random access container.
  //@{
public:

  // We need these using declarations to make the const versions visible.
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
    return _data + Base::size();
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
  operator[](const typename Base::size_type n)
  {
    return _data[n];
  }

  //! Fill the %array with the specified value.
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

  // We need these using declarations to make the const versions visible.
  using Base::operator();
  using Base::data;

  //! Array indexing.
  reference
  operator()(const typename Base::IndexList& indices)
  {
#ifdef STLIB_DEBUG
    for (typename Base::size_type n = 0; n != Base::Dimension; ++n) {
      assert(indices[n] < Base::extents()[n]);
    }
#endif
    return _data[Base::arrayIndex(indices)];
  }

  //! Array indexing.
  /*! \note The array dimension must be one in order to use this function. */
  reference
  operator()(const typename Base::Index i0)
  {
#ifdef STLIB_DEBUG
    assert(i0 < Base::extents()[0]);
#endif
    return _data[Base::arrayIndex(i0)];
  }

  //! Array indexing.
  /*! \note The array dimension must be two in order to use this function. */
  reference
  operator()(const typename Base::Index i0, const typename Base::Index i1)
  {
#ifdef STLIB_DEBUG
    assert(i0 < Base::extents()[0] && i1 < Base::extents()[1]);
#endif
    return _data[Base::arrayIndex(i0, i1)];
  }

  //! Array indexing.
  /*! \note The array dimension must be three in order to use this function. */
  reference
  operator()(const typename Base::Index i0, const typename Base::Index i1,
             const typename Base::Index i2)
  {
#ifdef STLIB_DEBUG
    assert(i0 < Base::extents()[0] && i1 < Base::extents()[1] &&
           i2 < Base::extents()[2]);
#endif
    return _data[Base::arrayIndex(i0, i1, i2)];
  }

  //! Return a pointer to the beginning of the data.
  pointer
  data()
  {
    return _data;
  }

protected:

  //! Set the data pointer.
  void
  setData(pointer data)
  {
    _data = data;
    Base::_constData = data;
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup SimpleMultiArrayRefAssignmentOperatorsScalar Assignment Operators with Scalar Operand
//@{

//! Array-scalar addition.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator+=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value);

//! Array-scalar subtraction.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator-=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value);

//! Array-scalar multiplication.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator*=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value);

//! Array-scalar division.
/*!
  \relates SimpleMultiArrayRef
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator/=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value);

//! Array-scalar modulus.
/*!
  \relates SimpleMultiArrayRef
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator%=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value);

//! Left shift.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator<<=(SimpleMultiArrayRef<_T, _Dimension>& x, int offset);

//! Right shift.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
SimpleMultiArrayRef<_T, _Dimension>&
operator>>=(SimpleMultiArrayRef<_T, _Dimension>& x, int offset);

//@}
//----------------------------------------------------------------------------
//! \defgroup SimpleMultiArrayRefFile SimpleMultiArrayRef File I/O
//@{

//! Read the %array extents and elements.
/*! \relates SimpleMultiArrayRef */
template<typename _T, std::size_t _Dimension>
std::istream&
operator>>(std::istream& in, SimpleMultiArrayRef<_T, _Dimension>& x);

//@}

} // namespace container
}

#define __container_SimpleMultiArrayRef_ipp__
#include "stlib/container/SimpleMultiArrayRef.ipp"
#undef __container_SimpleMultiArrayRef_ipp__

#endif

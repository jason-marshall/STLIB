// -*- C++ -*-

/*!
  \file stlib/container/EquilateralArrayImp.h
  \brief A multi-array that has equal extents in each dimension.
*/

#if !defined(__container_EquilateralArrayImp_h__)
#define __container_EquilateralArrayImp_h__

#include "stlib/ext/array.h"
#include "stlib/numerical/constants/Exponentiation.h"

#include <boost/call_traits.hpp>

namespace stlib
{
namespace container
{

//! A multi-array that has equal extents in each dimension.
/*!
  \param _T The value type.
  \param _D The dimension
  \param N The extent in each dimension.
  \param _Base The base class that implements the std::array interface.
*/
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
class EquilateralArrayImp :
  public _Base
{
  //
  // Constants.
  //
public:

  //! The total number of elements.
  BOOST_STATIC_CONSTEXPR std::size_t Size =
    numerical::Exponentiation<std::size_t, N, _D>::Result;

  //
  // Types.
  //
private:

  typedef _Base Base;

public:

  //! An array index is the same as the size type.
  typedef typename Base::size_type Index;
  //! A list of indices.
  typedef std::array<Index, _D> IndexList;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default constructor, copy constructor, assignment operator, and
    destructor. */
  // @{
public:

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  // GCC 4.0 does not properly implement the data() member for array.
#if __GNUC__ == 4 && __GNUC_MINOR__ == 0
  //! Return a const iterator to the data.
  typename Base::const_iterator
  data() const
  {
    return Base::begin();
  }
  //! Return an iterator to the data.
  typename Base::iterator
  data()
  {
    return Base::begin();
  }
#endif

  //! Return the extents.
  std::array<typename Base::size_type, _D>
  extents() const
  {
    return ext::filled_array<std::array<typename Base::size_type, _D> >
           (N);
  }

  //! Return a const reference to the specified element in the 0-D array.
  typename Base::const_reference
  operator()() const
  {
    static_assert(_D == 0, "Bad dimension.");
    return Base::operator[](0);
  }

  //! Return a const reference to the specified element in the 0-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 0>& /*i*/) const
  {
    return operator()();
  }

  //! Return a const reference to the specified element in the 1-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0) const
  {
    static_assert(_D == 1, "Bad dimension.");
    return Base::operator[](i0);
  }

  //! Return a const reference to the specified element in the 1-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 1>& i) const
  {
    return operator()(i[0]);
  }

  //! Return a const reference to the specified element in the 2-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1) const
  {
    static_assert(_D == 2, "Bad dimension.");
    return Base::operator[](i0 + i1 * N);
  }

  //! Return a const reference to the specified element in the 2-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 2>& i) const
  {
    return operator()(i[0], i[1]);
  }

  //! Return a const reference to the specified element in the 3-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2)
  const
  {
    static_assert(_D == 3, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N);
  }

  //! Return a const reference to the specified element in the 3-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 3>& i) const
  {
    return operator()(i[0], i[1], i[2]);
  }

  //! Return a const reference to the specified element in the 4-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3) const
  {
    static_assert(_D == 4, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N);
  }

  //! Return a const reference to the specified element in the 4-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 4>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3]);
  }

  //! Return a const reference to the specified element in the 5-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4) const
  {
    static_assert(_D == 5, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 5-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 5>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4]);
  }

  //! Return a const reference to the specified element in the 6-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5) const
  {
    static_assert(_D == 6, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 6-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 6>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5]);
  }

  //! Return a const reference to the specified element in the 7-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6) const
  {
    static_assert(_D == 7, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 7-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 7>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
  }

  //! Return a const reference to the specified element in the 8-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7) const
  {
    static_assert(_D == 8, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 8-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 8>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
  }

  //! Return a const reference to the specified element in the 9-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7,
             const typename Base::size_type i8) const
  {
    static_assert(_D == 9, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N +
                            i8 * N * N * N * N * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 9-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 9>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]);
  }

  //! Return a const reference to the specified element in the 10-D array.
  typename Base::const_reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7,
             const typename Base::size_type i8,
             const typename Base::size_type i9) const
  {
    static_assert(_D == 10, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N +
                            i8 * N * N * N * N * N * N * N * N +
                            i9 * N * N * N * N * N * N * N * N * N);
  }

  //! Return a const reference to the specified element in the 10-D array.
  template<typename _Integer>
  typename Base::const_reference
  operator()(const std::array<_Integer, 10>& i) const
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8],
                      i[9]);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Return a reference to the specified element in the 0-D array.
  typename Base::reference
  operator()()
  {
    static_assert(_D == 0, "Bad dimension.");
    return Base::operator[](0);
  }

  //! Return a reference to the specified element in the 0-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 0>& /*i*/)
  {
    return operator()();
  }

  //! Return a reference to the specified element in the 1-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0)
  {
    static_assert(_D == 1, "Bad dimension.");
    return Base::operator[](i0);
  }

  //! Return a reference to the specified element in the 1-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 1>& i)
  {
    return operator()(i[0]);
  }

  //! Return a reference to the specified element in the 2-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1)
  {
    static_assert(_D == 2, "Bad dimension.");
    return Base::operator[](i0 + i1 * N);
  }

  //! Return a reference to the specified element in the 2-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 2>& i)
  {
    return operator()(i[0], i[1]);
  }

  //! Return a reference to the specified element in the 3-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2)
  {
    static_assert(_D == 3, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N);
  }

  //! Return a reference to the specified element in the 3-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 3>& i)
  {
    return operator()(i[0], i[1], i[2]);
  }

  //! Return a reference to the specified element in the 4-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3)
  {
    static_assert(_D == 4, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N);
  }

  //! Return a reference to the specified element in the 4-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 4>& i)
  {
    return operator()(i[0], i[1], i[2], i[3]);
  }

  //! Return a reference to the specified element in the 5-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4)
  {
    static_assert(_D == 5, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N);
  }

  //! Return a reference to the specified element in the 5-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 5>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4]);
  }

  //! Return a reference to the specified element in the 6-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5)
  {
    static_assert(_D == 6, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N);
  }

  //! Return a reference to the specified element in the 6-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 6>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5]);
  }

  //! Return a reference to the specified element in the 7-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6)
  {
    static_assert(_D == 7, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N);
  }

  //! Return a reference to the specified element in the 7-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 7>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
  }

  //! Return a reference to the specified element in the 8-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7)
  {
    static_assert(_D == 8, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N);
  }

  //! Return a reference to the specified element in the 8-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 8>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
  }

  //! Return a reference to the specified element in the 9-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7,
             const typename Base::size_type i8)
  {
    static_assert(_D == 9, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N +
                            i8 * N * N * N * N * N * N * N * N);
  }

  //! Return a reference to the specified element in the 9-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 9>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]);
  }

  //! Return a reference to the specified element in the 10-D array.
  typename Base::reference
  operator()(const typename Base::size_type i0,
             const typename Base::size_type i1,
             const typename Base::size_type i2,
             const typename Base::size_type i3,
             const typename Base::size_type i4,
             const typename Base::size_type i5,
             const typename Base::size_type i6,
             const typename Base::size_type i7,
             const typename Base::size_type i8,
             const typename Base::size_type i9)
  {
    static_assert(_D == 10, "Bad dimension.");
    return Base::operator[](i0 + i1 * N + i2 * N * N + i3 * N * N * N +
                            i4 * N * N * N * N + i5 * N * N * N * N * N +
                            i6 * N * N * N * N * N * N +
                            i7 * N * N * N * N * N * N * N +
                            i8 * N * N * N * N * N * N * N * N +
                            i9 * N * N * N * N * N * N * N * N * N);
  }

  //! Return a reference to the specified element in the 10-D array.
  template<typename _Integer>
  typename Base::reference
  operator()(const std::array<_Integer, 10>& i)
  {
    return operator()(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8],
                      i[9]);
  }

  // @}
};


//----------------------------------------------------------------------------
//! \defgroup EquilateralArrayAssignmentOperatorsScalar Assignment Operators with Scalar Operand
//@{


//! Array-scalar addition.
/*! \relates EquilateralArrayImp */
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator+=(EquilateralArrayImp<_T, _D, N, _B>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i += value;
  }
  return x;
}


//! Array-scalar subtraction.
/*! \relates EquilateralArrayImp */
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator-=(EquilateralArrayImp<_T, _D, N, _B>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i -= value;
  }
  return x;
}


//! Array-scalar multiplication.
/*! \relates EquilateralArrayImp */
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator*=(EquilateralArrayImp<_T, _D, N, _B>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i *= value;
  }
  return x;
}


//! Array-scalar division.
/*!
  \relates EquilateralArrayImp
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator/=(EquilateralArrayImp<_T, _D, N, _B>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i /= value;
  }
  return x;
}


//! Array-scalar modulus.
/*!
  \relates EquilateralArrayImp
  \note This does not check for division by zero as the value type may not be
  as number type.
*/
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator%=(EquilateralArrayImp<_T, _D, N, _B>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i %= value;
  }
  return x;
}


//! Left shift.
/*! \relates EquilateralArrayImp */
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator<<=(EquilateralArrayImp<_T, _D, N, _B>& x, const int offset)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i <<= offset;
  }
  return x;
}


//! Right shift.
/*! \relates EquilateralArrayImp */
template<typename _T, std::size_t _D, std::size_t N, typename _B>
inline
EquilateralArrayImp<_T, _D, N, _B>&
operator>>=(EquilateralArrayImp<_T, _D, N, _B>& x, const int offset)
{
  typedef typename EquilateralArrayImp<_T, _D, N, _B>::iterator iterator;
  for (iterator i = x.begin(); i != x.end(); ++i) {
    *i >>= offset;
  }
  return x;
}


//@}


} // namespace container
}

#endif

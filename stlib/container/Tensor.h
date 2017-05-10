// -*- C++ -*-

/*!
  \file Tensor.h
  \brief A class for a static array of arrays.
*/

#if !defined(__container_Tensor_h__)
#define __container_Tensor_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace container
{

//! A tensor of degree 1.
/*!
  \param _T The value type.
  \param N0 The extent in the first (and only) dimension.

  \note Tensor1 inherits from \c std::array<_T,N0> so it is not P.O.D.
*/
template<typename _T, std::size_t N0>
class Tensor1 : public std::array<_T, N0>
{
  //
  // Public Types.
  //

public:

  //! The base class.
  typedef std::array<_T, N0> Base;
  //! The value type.
  typedef typename Base::value_type value_type;
  //! Reference to the value type.
  typedef typename Base::reference reference;
  //! Constant reference to the value type.
  typedef typename Base::const_reference const_reference;
  //! Iterator in the container.
  typedef typename Base::iterator iterator;
  //! Constant iterator in the container.
  typedef typename Base::const_iterator const_iterator;
  //! The size type.
  typedef typename Base::size_type size_type;
  //! The pointer difference type.
  typedef typename Base::difference_type difference_type;
  //! Reverse iterator.
  typedef typename Base::reverse_iterator reverse_iterator;
  //! Constant reverse iterator.
  typedef typename Base::const_reverse_iterator const_reverse_iterator;

  //
  // Use from the base class.
  //
public:

  using Base::swap;
  using Base::begin;
  using Base::end;
  using Base::rbegin;
  using Base::rend;
  using Base::size;
  using Base::max_size;
  using Base::empty;
  using Base::operator[];
  using Base::at;
  using Base::front;
  using Base::back;
  using Base::data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{
public:

  //! Default constructor.
  Tensor1() :
    Base()
  {
  }

  //! Construct from a \c std::array of possibly different value type.
  template<typename _T2>
  Tensor1(const std::array<_T2, N0>& data) :
    Base()
  {
    std::copy(data.begin(), data.end(), begin());
  }

  //! Construct from a range of data.
  template<typename _InputIterator>
  Tensor1(_InputIterator start, _InputIterator finish) :
    Base()
  {
    // Assume that the range has the correct number of elements.
    std::copy(start, finish, begin());
  }

  //! Construct from a pointer to data.
  template<typename _T2>
  Tensor1(const _T2* data) :
    Base()
  {
    std::copy(data, data + size(), begin());
  }

  // Use the default copy constructor, assignment operator, and destructor.

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the extents.
  std::array<size_type, 1>
  extents() const
  {
    return std::array<size_type, 1>{{size()}};
  }

  //! Return a const reference to the specified element.
  const_reference
  operator()(const size_type i) const
  {
    return operator[](i);
  }

  //! Return a const reference to the specified element.
  template<typename _Integer>
  const_reference
  operator()(const std::array<_Integer, 1>& i) const
  {
    return operator[](i[0]);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Return a reference to the specified element.
  reference
  operator()(const size_type i)
  {
    return operator[](i);
  }

  //! Return a reference to the specified element.
  template<typename _Integer>
  reference
  operator()(const std::array<_Integer, 1>& i)
  {
    return operator[](i[0]);
  }

  // @}
};


//! A tensor of degree 2.
/*!
  \param _T The value type.
  \param N0 The extent in the first dimension.
  \param N1 The extent in the second dimension.

  The data is laid out in column-major order. That is, the first dimension
  varies fastest.

  \note Tensor2 inherits from \c std::array<_T,N0*N1> so it is not P.O.D.
*/
template<typename _T, std::size_t N0, std::size_t N1>
class Tensor2 : public std::array<_T, N0* N1>
{
  //
  // Public Types.
  //

public:

  //! The base class.
  typedef std::array<_T, N0* N1> Base;
  //! The value type.
  typedef typename Base::value_type value_type;
  //! Reference to the value type.
  typedef typename Base::reference reference;
  //! Constant reference to the value type.
  typedef typename Base::const_reference const_reference;
  //! Iterator in the container.
  typedef typename Base::iterator iterator;
  //! Constant iterator in the container.
  typedef typename Base::const_iterator const_iterator;
  //! The size type.
  typedef typename Base::size_type size_type;
  //! The pointer difference type.
  typedef typename Base::difference_type difference_type;
  //! Reverse iterator.
  typedef typename Base::reverse_iterator reverse_iterator;
  //! Constant reverse iterator.
  typedef typename Base::const_reverse_iterator const_reverse_iterator;

  //
  // Use from the base class.
  //
public:

  using Base::swap;
  using Base::begin;
  using Base::end;
  using Base::rbegin;
  using Base::rend;
  using Base::size;
  using Base::max_size;
  using Base::empty;
  using Base::operator[];
  using Base::at;
  using Base::front;
  using Base::back;
  using Base::data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{
public:

  //! Default constructor.
  Tensor2() :
    Base()
  {
  }

  //! Construct from a \c std::array of possibly different value type.
  template<typename _T2>
  Tensor2(const std::array<_T2, N0* N1>& data) :
    Base()
  {
    std::copy(data.begin(), data.end(), begin());
  }

  //! Construct from a range of data.
  template<typename _InputIterator>
  Tensor2(_InputIterator start, _InputIterator finish) :
    Base()
  {
    // Assume that the range has the correct number of elements.
    std::copy(start, finish, begin());
  }

  //! Construct from a pointer to data.
  template<typename _T2>
  Tensor2(const _T2* data) :
    Base()
  {
    std::copy(data, data + size(), begin());
  }

  // Use the default copy constructor, assignment operator, and destructor.

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the extents.
  std::array<size_type, 2>
  extents() const
  {
    return std::array<size_type, 2>{{N0, N1}};
  }

  //! Return a const reference to the specified element.
  const_reference
  operator()(const size_type i, const size_type j) const
  {
    return operator[](i + j * N0);
  }

  //! Return a const reference to the specified element.
  template<typename _Integer>
  const_reference
  operator()(const std::array<_Integer, 2>& i) const
  {
    return operator()(i[0], i[1]);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Return a reference to the specified element.
  reference
  operator()(const size_type i, const size_type j)
  {
    return operator[](i + j * N0);
  }

  //! Return a reference to the specified element.
  template<typename _Integer>
  reference
  operator()(const std::array<_Integer, 2>& i)
  {
    return operator()(i[0], i[1]);
  }

  // @}
};


//! A tensor of degree 3.
/*!
  \param _T The value type.
  \param N0 The extent in the first dimension.
  \param N1 The extent in the second dimension.
  \param N2 The extent in the third dimension.

  The data is laid out in column-major order. That is, the first dimension
  varies fastest.

  \note Tensor3 inherits from \c std::array<_T,N0*N1*N2> so it is
  not P.O.D.
*/
template<typename _T, std::size_t N0, std::size_t N1, std::size_t N2>
class Tensor3 : public std::array<_T, N0* N1* N2>
{
  //
  // Public Types.
  //

public:

  //! The base class.
  typedef std::array<_T, N0* N1* N2> Base;
  //! The value type.
  typedef typename Base::value_type value_type;
  //! Reference to the value type.
  typedef typename Base::reference reference;
  //! Constant reference to the value type.
  typedef typename Base::const_reference const_reference;
  //! Iterator in the container.
  typedef typename Base::iterator iterator;
  //! Constant iterator in the container.
  typedef typename Base::const_iterator const_iterator;
  //! The size type.
  typedef typename Base::size_type size_type;
  //! The pointer difference type.
  typedef typename Base::difference_type difference_type;
  //! Reverse iterator.
  typedef typename Base::reverse_iterator reverse_iterator;
  //! Constant reverse iterator.
  typedef typename Base::const_reverse_iterator const_reverse_iterator;

  //
  // Use from the base class.
  //
public:

  using Base::swap;
  using Base::begin;
  using Base::end;
  using Base::rbegin;
  using Base::rend;
  using Base::size;
  using Base::max_size;
  using Base::empty;
  using Base::operator[];
  using Base::at;
  using Base::front;
  using Base::back;
  using Base::data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{
public:

  //! Default constructor.
  Tensor3() :
    Base()
  {
  }

  //! Construct from a \c std::array of possibly different value type.
  template<typename _T2>
  Tensor3(const std::array<_T2, N0* N1* N2>& data) :
    Base()
  {
    std::copy(data.begin(), data.end(), begin());
  }

  //! Construct from a range of data.
  template<typename _InputIterator>
  Tensor3(_InputIterator start, _InputIterator finish) :
    Base()
  {
    // Assume that the range has the correct number of elements.
    std::copy(start, finish, begin());
  }

  //! Construct from a pointer to data.
  template<typename _T2>
  Tensor3(const _T2* data) :
    Base()
  {
    std::copy(data, data + size(), begin());
  }

  // Use the default copy constructor, assignment operator, and destructor.

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the extents.
  std::array<size_type, 3>
  extents() const
  {
    return std::array<size_type, 3>{{N0, N1, N2}};
  }

  //! Return a const reference to the specified element.
  const_reference
  operator()(const size_type i, const size_type j, const size_type k) const
  {
    return operator[](i + j * N0 + k * N0 * N1);
  }

  //! Return a const reference to the specified element.
  template<typename _Integer>
  const_reference
  operator()(const std::array<_Integer, 3>& i) const
  {
    return operator()(i[0], i[1], i[2]);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Return a reference to the specified element.
  reference
  operator()(const size_type i, const size_type j, const size_type k)
  {
    return operator[](i + j * N0 + k * N0 * N1);
  }

  //! Return a reference to the specified element.
  template<typename _Integer>
  reference
  operator()(const std::array<_Integer, 3>& i)
  {
    return operator()(i[0], i[1], i[2]);
  }

  // @}
};


} // namespace container
}

#endif

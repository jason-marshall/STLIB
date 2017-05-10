// -*- C++ -*-

/*!
  \file stlib/container/ViewIterator.h
  \brief An iterator for a view of an array.
*/

#if !defined(__container_ViewIterator_h__)
#define __container_ViewIterator_h__

#include "stlib/container/IndexTypes.h"

#include <boost/mpl/if.hpp>

#include <iterator>

namespace stlib
{
namespace container
{

//! An iterator for a view of an array.
template<typename _Array, bool _IsConst>
class
  ViewIterator
{
  //
  // Types.
  //
private:

  typedef IndexTypes Types;

public:

  //! The multi-array type.
  typedef _Array Array;
  //! Reference to the multi-array.
  typedef typename boost::mpl::if_c<_IsConst, const Array&, Array&>::type
  ArrayReference;
  //! Pointer to the multi-array.
  typedef typename boost::mpl::if_c<_IsConst, const Array*, Array*>::type
  ArrayPointer;
  //! The size type.
  typedef typename Types::size_type size_type;
  //! An array index is a signed integer.
  typedef typename Types::Index Index;

  // Iterator types.

  //! Random access iterator category.
  typedef std::random_access_iterator_tag iterator_category;
  //! Value type.
  typedef typename Array::value_type value_type;
  //! Pointer difference type.
  typedef typename Types::difference_type difference_type;
  //! Reference to the value type.
  typedef typename boost::mpl::if_c<_IsConst, const value_type&, value_type&>::
  type reference;
  //! Pointer to the value type.
  typedef typename boost::mpl::if_c<_IsConst, const value_type*, value_type*>::
  type pointer;

  //
  // Member data.
  //
private:

  //! Pointer in the array.
  pointer _iterator;
  //! Pointer to the multi-array.
  ArrayPointer _array;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Return an iterator to the beginning of the index range.
  static
  ViewIterator
  begin(ArrayReference array);

  //! Return an iterator to the end of the index range.
  static
  ViewIterator
  end(ArrayReference array);

  // The default copy constructor, assignment operator and destructor are fine.

  //! Copy constructor from non-const.
  template<bool _IsConst2>
  ViewIterator(const ViewIterator<Array, _IsConst2>& other);

  //! Assignment operator from non-const.
  template<bool _IsConst2>
  ViewIterator&
  operator=(const ViewIterator<Array, _IsConst2>& other);

private:

  //! Default constructor. Default instantiation.
  ViewIterator() :
    _iterator(),
    _array()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The index.
  Index
  index() const
  {
    return _array->base() + rank();
  }

  //! The multi-array.
  ArrayPointer
  array() const
  {
    return _array;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Validity.
  //@{
private:

  //! Return true if the iterator is valid.
  /*!
    It's valid if it is in the range [begin(), end()).
  */
  bool
  isValid() const
  {
    return _array->data() <= _iterator &&
           _iterator < _array->data() + _array->size();
  }

  //! Return true if the iterator is at the beginning.
  bool
  isBegin() const
  {
    return _iterator == _array->data();
  }

  //! Return true if the iterator is at the end.
  bool
  isEnd() const
  {
    return _iterator == _array->data() + _array->size();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Forward iterator requirements.
  //@{
public:

  reference
  operator*() const
  {
    return *_iterator;
  }

  pointer
  operator->() const
  {
    return _iterator;
  }

  //! Pre-increment.
  ViewIterator&
  operator++();

  //! Post-increment.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  ViewIterator
  operator++(int);

  //@}
  //--------------------------------------------------------------------------
  //! \name Bidirectional iterator requirements.
  //@{
public:

  //! Pre-decrement.
  ViewIterator&
  operator--();

  //! Post-decrement.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  ViewIterator
  operator--(int);

  //@}
  //--------------------------------------------------------------------------
  //! \name Random access iterator requirements.
  //@{
public:

  //! Iterator indexing.
  /*!
    \warning This function is inefficient.
  */
  value_type
  operator[](const difference_type n) const
  {
    return *(*this + n);
  }

  ViewIterator&
  operator+=(const difference_type n)
  {
    _iterator += n * _array->stride();
    return *this;
  }

  ViewIterator
  operator+(const difference_type n) const
  {
    ViewIterator tmp(*this);
    tmp += n;
    return tmp;
  }

  ViewIterator&
  operator-=(const difference_type n)
  {
    _iterator -= n * _array->stride();
    return *this;
  }

  ViewIterator
  operator-(const difference_type n) const
  {
    ViewIterator tmp(*this);
    tmp -= n;
    return tmp;
  }

  Index
  rank() const
  {
    return (_iterator - _array->data()) / _array->stride();
  }

  pointer
  base() const
  {
    return _iterator;
  }
  //@}
};

//---------------------------------------------------------------------------
// Equality.

//! Return true if the iterators are equal.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator==(const ViewIterator<_Array, _IsConst1>& x,
           const ViewIterator<_Array, _IsConst2>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.array() == y.array());
#endif
  return x.base() == y.base();
}

//! Return true if they are not equal.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator!=(const ViewIterator<_Array, _IsConst1>& x,
           const ViewIterator<_Array, _IsConst2>& y)
{
  return !(x == y);
}


//! Return true if the first precedes the second.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator<(const ViewIterator<_Array, _IsConst1>& x,
          const ViewIterator<_Array, _IsConst2>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.array() == y.array());
#endif
  return x.base() < y.base();
}

//! Return y < x.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator>(const ViewIterator<_Array, _IsConst1>& x,
          const ViewIterator<_Array, _IsConst2>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator<=(const ViewIterator<_Array, _IsConst1>& x,
           const ViewIterator<_Array, _IsConst2>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
bool
operator>=(const ViewIterator<_Array, _IsConst1>& x,
           const ViewIterator<_Array, _IsConst2>& y)
{
  return !(x < y);
}

//! Return the difference between the two iterators.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst1, bool _IsConst2>
inline
typename ViewIterator<_Array, _IsConst1>::difference_type
operator-(const ViewIterator<_Array, _IsConst1>& x,
          const ViewIterator<_Array, _IsConst2>& y)
{
  return x.base() - y.base();
}

//! Advance the iterator.
/*! \relates ViewIterator */
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>
operator+(const typename ViewIterator<_Array, _IsConst>::
          difference_type& n,
          const ViewIterator<_Array, _IsConst>& x)
{
  return x + n;
}

} // namespace container
}

#define __container_ViewIterator_ipp__
#include "stlib/container/ViewIterator.ipp"
#undef __container_ViewIterator_ipp__

#endif

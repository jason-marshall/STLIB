// -*- C++ -*-

/*!
  \file stlib/container/MultiViewIterator.h
  \brief An iterator for a view of an %array.
*/

#if !defined(__container_MultiViewIterator_h__)
#define __container_MultiViewIterator_h__

#include "stlib/container/MultiIndexTypes.h"

#include <boost/mpl/if.hpp>

#include <iterator>

namespace stlib
{
namespace container
{

//! An iterator for a view of an %array.
template<typename _MultiArray, bool _IsConst>
class
  MultiViewIterator
{
  //
  // Constants.
  //
public:

  //! The number of dimensions.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _MultiArray::Dimension;

  //
  // Types.
  //
private:

  typedef MultiIndexTypes<Dimension> Types;

public:

  //! The multi-array type.
  typedef _MultiArray MultiArray;
  //! Reference to the multi-array.
  typedef typename boost::mpl::if_c<_IsConst, const MultiArray&,
          MultiArray& >::type MultiArrayReference;
  //! Pointer to the multi-array.
  typedef typename boost::mpl::if_c<_IsConst, const MultiArray*,
          MultiArray* >::type MultiArrayPointer;
  //! The size type.
  typedef typename Types::size_type size_type;
  //! An %array index is a signed integer.
  typedef typename Types::Index Index;
  //! A list of indices.
  typedef typename Types::IndexList IndexList;
  //! A list of sizes.
  typedef typename Types::SizeList SizeList;
  //! The storage order.
  typedef typename Types::Storage Storage;

  // Iterator types.

  //! Random access iterator category.
  typedef std::random_access_iterator_tag iterator_category;
  //! Value type.
  typedef typename MultiArray::value_type value_type;
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

  //! An index list.
  IndexList _indexList;
  //! The rank of the index list.
  Index _rank;
  //! Pointer in the %array.
  pointer _iterator;
  //! Pointer to the multi-array.
  MultiArrayPointer _array;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Return an iterator to the beginning of the index range.
  static
  MultiViewIterator
  begin(MultiArrayReference array);

  //! Return an iterator to the end of the index range.
  static
  MultiViewIterator
  end(MultiArrayReference array);

  // The default copy constructor, assignment operator and destructor are fine.

  //! Copy constructor from non-const.
  template<bool _IsConst2>
  MultiViewIterator(const MultiViewIterator<MultiArray, _IsConst2>& other);

  //! Assignment operator from non-const.
  template<bool _IsConst2>
  MultiViewIterator&
  operator=(const MultiViewIterator<MultiArray, _IsConst2>& other);

private:

  //! Default constructor. Uninitialized data.
  MultiViewIterator()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The index list.
  IndexList
  indexList() const
  {
    return _indexList;
  }

  //! The multi-array.
  MultiArrayPointer
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
  isValid() const;

  //! Return true if the iterator is at the beginning.
  bool
  isBegin() const;

  //! Return true if the iterator is at the end.
  bool
  isEnd() const;

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
  MultiViewIterator&
  operator++();

  //! Post-increment.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  MultiViewIterator
  operator++(int);

  //@}
  //--------------------------------------------------------------------------
  //! \name Bidirectional iterator requirements.
  //@{
public:

  //! Pre-decrement.
  MultiViewIterator&
  operator--();

  //! Post-decrement.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  MultiViewIterator
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

  MultiViewIterator&
  operator+=(const difference_type n)
  {
    _rank += n;
    update();
    return *this;
  }

  MultiViewIterator
  operator+(const difference_type n) const
  {
    MultiViewIterator tmp(*this);
    tmp += n;
    return tmp;
  }

  MultiViewIterator&
  operator-=(const difference_type n)
  {
    _rank -= n;
    update();
    return *this;
  }

  MultiViewIterator
  operator-(const difference_type n) const
  {
    MultiViewIterator tmp(*this);
    tmp -= n;
    return tmp;
  }

  Index
  rank() const
  {
    return _rank;
  }

  pointer
  base() const
  {
    return _iterator;
  }

private:

  //! Calculate the index list and the base iterator from the rank.
  void
  update();

  //@}
};

//---------------------------------------------------------------------------
// Equality.

//! Return true if the iterators are equal.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator==(const MultiViewIterator<_MultiArray, _IsConst1>& x,
           const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.array() == y.array());
#endif
  return x.rank() == y.rank();
}

//! Return true if they are not equal.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator!=(const MultiViewIterator<_MultiArray, _IsConst1>& x,
           const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
  return !(x == y);
}


//! Return true if the first precedes the second.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator<(const MultiViewIterator<_MultiArray, _IsConst1>& x,
          const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.array() == y.array());
#endif
  return x.rank() < y.rank();
}

//! Return y < x.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator>(const MultiViewIterator<_MultiArray, _IsConst1>& x,
          const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator<=(const MultiViewIterator<_MultiArray, _IsConst1>& x,
           const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
bool
operator>=(const MultiViewIterator<_MultiArray, _IsConst1>& x,
           const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
  return !(x < y);
}

//! Return the difference between the two iterators.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst1, bool _IsConst2>
inline
typename MultiViewIterator<_MultiArray, _IsConst1>::difference_type
operator-(const MultiViewIterator<_MultiArray, _IsConst1>& x,
          const MultiViewIterator<_MultiArray, _IsConst2>& y)
{
  return x.rank() - y.rank();
}

//! Advance the iterator.
/*! \relates MultiViewIterator */
template<typename _MultiArray, bool _IsConst>
inline
MultiViewIterator<_MultiArray, _IsConst>
operator+(const typename MultiViewIterator<_MultiArray, _IsConst>::
          difference_type& n,
          const MultiViewIterator<_MultiArray, _IsConst>& x)
{
  return x + n;
}

} // namespace container
}

#define __container_MultiViewIterator_ipp__
#include "stlib/container/MultiViewIterator.ipp"
#undef __container_MultiViewIterator_ipp__

#endif

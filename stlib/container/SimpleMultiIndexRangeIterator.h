// -*- C++ -*-

/*!
  \file stlib/container/SimpleMultiIndexRangeIterator.h
  \brief An index range iterator.
*/

#if !defined(__container_SimpleMultiIndexRangeIterator_h__)
#define __container_SimpleMultiIndexRangeIterator_h__

#include "stlib/container/SimpleMultiIndexRange.h"

#include <iterator>

namespace stlib
{
namespace container
{

//! An index range iterator.
template<std::size_t _Dimension>
class SimpleMultiIndexRangeIterator
{
  //
  // Constants.
  //
public:

  //! The number of dimensions.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;

  //
  // Types.
  //
public:

  //! An index range.
  typedef SimpleMultiIndexRange<_Dimension> Range;
  //! An array index type is \c std::size_t.
  typedef std::size_t Index;
  //! A list of indices.
  typedef std::array<std::size_t, Dimension> IndexList;

  // Iterator types.

  //! Random access iterator category.
  typedef std::random_access_iterator_tag iterator_category;
  //! Value type.
  typedef IndexList value_type;
  //! Pointer difference type.
  typedef std::ptrdiff_t difference_type;
  //! Const reference to the value type.
  typedef const value_type& reference;
  //! Const pointer to the value type.
  typedef const value_type* pointer;

  //
  // Member data.
  //
private:

  //! An index list.
  IndexList _indexList;
  //! The rank of the index list.
  Index _rank;
  //! The index range.
  Range _range;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Return an iterator to the beginning of the index range.
  static
  SimpleMultiIndexRangeIterator
  begin(const Range& range);

  //! Return an iterator to the beginning of the index range.
  /*! Since the argument is the range extents, the bases are zero. */
  static
  SimpleMultiIndexRangeIterator
  begin(const IndexList& extents);

  //! Return an iterator to the end of the index range.
  static
  SimpleMultiIndexRangeIterator
  end(const Range& range);

  //! Return an iterator to the end of the index range.
  /*! Since the argument is the range extents, the bases are zero. */
  static
  SimpleMultiIndexRangeIterator
  end(const IndexList& extents);

private:

  //! Default constructor. Uninitialized memory.
  SimpleMultiIndexRangeIterator()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The index range.
  const Range&
  range() const
  {
    return _range;
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
    return _indexList;
  }

  pointer
  operator->() const
  {
    return &_indexList;
  }

  //! Pre-increment.
  SimpleMultiIndexRangeIterator&
  operator++();

  //! Post-increment.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  SimpleMultiIndexRangeIterator
  operator++(int);

  //@}
  //--------------------------------------------------------------------------
  //! \name Bidirectional iterator requirements.
  //@{
public:

  //! Pre-decrement.
  SimpleMultiIndexRangeIterator&
  operator--();

  //! Post-decrement.
  /*!
    \warning This function is inefficient. Use pre-increment instead.
  */
  SimpleMultiIndexRangeIterator
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

  SimpleMultiIndexRangeIterator&
  operator+=(const difference_type n)
  {
    _rank += n;
    calculateIndexList();
    return *this;
  }

  SimpleMultiIndexRangeIterator
  operator+(const difference_type n) const
  {
    SimpleMultiIndexRangeIterator tmp(*this);
    tmp += n;
    return tmp;
  }

  SimpleMultiIndexRangeIterator&
  operator-=(const difference_type n)
  {
    _rank -= n;
    calculateIndexList();
    return *this;
  }

  SimpleMultiIndexRangeIterator
  operator-(const difference_type n) const
  {
    SimpleMultiIndexRangeIterator tmp(*this);
    tmp -= n;
    return tmp;
  }

  Index
  base() const
  {
    return _rank;
  }

private:

  //! Calculate the index list from the rank.
  void
  calculateIndexList();

  //@}
};

//---------------------------------------------------------------------------
// Equality.

//! Return true if the iterators are equal.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator==(const SimpleMultiIndexRangeIterator<_Dimension>& x,
           const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.range() == y.range());
#endif
  return x.base() == y.base();
}

//! Return true if they are not equal.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator!=(const SimpleMultiIndexRangeIterator<_Dimension>& x,
           const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
  return !(x == y);
}


//! Return true if the first precedes the second.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator<(const SimpleMultiIndexRangeIterator<_Dimension>& x,
          const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
#ifdef STLIB_DEBUG
  // The must be iterators over the same index range.
  assert(x.range() == y.range());
#endif
  return x.base() < y.base();
}

//! Return y < x.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator>(const SimpleMultiIndexRangeIterator<_Dimension>& x,
          const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator<=(const SimpleMultiIndexRangeIterator<_Dimension>& x,
           const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
bool
operator>=(const SimpleMultiIndexRangeIterator<_Dimension>& x,
           const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
  return !(x < y);
}

//! Return the difference between the two iterators.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
typename SimpleMultiIndexRangeIterator<_Dimension>::difference_type
operator-(const SimpleMultiIndexRangeIterator<_Dimension>& x,
          const SimpleMultiIndexRangeIterator<_Dimension>& y)
{
  typedef typename SimpleMultiIndexRangeIterator<_Dimension>::difference_type
  difference_type;
  return difference_type(x.base()) - difference_type(y.base());
}

//! Advance the iterator.
/*! \relates SimpleMultiIndexRangeIterator */
template<std::size_t _Dimension>
inline
SimpleMultiIndexRangeIterator<_Dimension>
operator+(const typename
          SimpleMultiIndexRangeIterator<_Dimension>::difference_type& n,
          const SimpleMultiIndexRangeIterator<_Dimension>& x)
{
  return x + n;
}

} // namespace container
}

#define __container_SimpleMultiIndexRangeIterator_ipp__
#include "stlib/container/SimpleMultiIndexRangeIterator.ipp"
#undef __container_SimpleMultiIndexRangeIterator_ipp__

#endif

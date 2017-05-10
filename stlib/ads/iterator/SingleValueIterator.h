// -*- C++ -*-

/*!
  \file SingleValueIterator.h
  \brief An input iterator whose value is a constant.
*/

#if !defined(__SingleValueIterator_h__)
#define __SingleValueIterator_h__

#include <iterator>

namespace stlib
{
namespace ads
{

//! An input iterator whose value is a constant.
/*!
  Suppose that you need to input a sequence of whose elements have the
  same value. Instead of having to build a container and fill it with this
  values, you can use a SingleValueIterator. This iterator just stores a
  value, which one obtains with dereferencing.

  As an example, instead of using a container:

  \verbatim
  std::vector<double> x(100, 1.);
  foo(x.begin(), x.end()); \endverbatim

  one can use a SingleValueIterator:

  \verbatim
  SingleValueIterator<double> begin(1.);
  foo(begin, begin + 100); \endverbatim

  In some circumstances, one can make the code more compact by using
  makeSingleValueIterator().

  \verbatim
  std::vector<double> x(100);
  ...
  const bool isZero = std::equal(x.begin(), x.end(), makeSingleValueIterator<double>(0)); \endverbatim
*/
template<typename _T>
class SingleValueIterator :
  public std::iterator<std::random_access_iterator_tag,
  _T,
  std::ptrdiff_t,
  const _T*,
  const _T&>
{
  //
  // Types.
  //
private:

  typedef std::iterator<std::random_access_iterator_tag,
          _T,
          std::ptrdiff_t,
          const _T*,
          const _T&> Base;

  //
  // Member data.
  //
private:

  typename Base::value_type _value;
  typename Base::pointer _base;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   We use the synthesized copy constructor, assignment operator,
   and destructor. */
  //@{
public:

  // Default constructor.  The value is initialized to zero.
  SingleValueIterator() :
    _value(0) {}

  //! Construct and initialize the value.
  SingleValueIterator(const typename Base::value_type x) :
    _value(x),
    _base(0) {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Forward iterator requirements
  //@{
public:

  //! Dereference.
  typename Base::reference
  operator*() const
  {
    // Return a reference to the value.
    return _value;
  }

  //! Pointer dereference.
  typename Base::pointer
  operator->() const
  {
    // Return a pointer to the value.
    return &_value;
  }

  //! Pre-increment.
  SingleValueIterator&
  operator++()
  {
    ++_base;
    return *this;
  }

  //! Post-increment.
  /*!
    \note This is not efficient.  If possible, use the pre-increment operator
    instead.
  */
  SingleValueIterator
  operator++(int)
  {
    SingleValueIterator x(*this);
    ++*this;
    return x;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Bidirectional iterator requirements
  //@{
public:

  //! Pre-decrement.
  SingleValueIterator&
  operator--()
  {
    --_base;
    return *this;
  }

  //! Post-decrement.
  /*!
    \note This is not efficient.  If possible, use the pre-decrement operator
    instead.
  */
  SingleValueIterator
  operator--(int)
  {
    SingleValueIterator x(*this);
    --*this;
    return x;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Random access iterator requirements
  //@{
public:

  //! Iterator indexing.
  typename Base::value_type
  operator[](const typename Base::difference_type /*n*/)
  {
    // Return the value offset by n.
    return _value;
  }

  //! Positive offseting.
  SingleValueIterator&
  operator+=(const typename Base::difference_type n)
  {
    _base += n;
    return *this;
  }

  //! Positive offseting.
  /*!
    \note This is not efficient.  If possible, use \c += instead.
  */
  SingleValueIterator
  operator+(const typename Base::difference_type n) const
  {
    SingleValueIterator x(*this);
    x += n;
    return x;
  }

  //! Negative offseting.
  SingleValueIterator&
  operator-=(const typename Base::difference_type n)
  {
    _base -= n;
    return *this;
  }

  //! Negative offseting.
  /*!
    \note This is not efficient.  If possible, use \c -= instead.
  */
  SingleValueIterator
  operator-(const typename Base::difference_type n) const
  {
    SingleValueIterator x(*this);
    x -= n;
    return x;
  }

  //! Return the base pointer.
  typename Base::pointer
  base() const
  {
    return _base;
  }

  //@}
};

//
// Convenience functions.
//

//! Convenience function for instantiating an SingleValueIterator.
/*! \relates SingleValueIterator */
template<typename _T>
SingleValueIterator<_T>
makeSingleValueIterator(const _T x)
{
  SingleValueIterator<_T> i(x);
  return i;
}

//
// Forward iterator requirements
//

//! Return true if the iterators have the same base.
/*! \relates SingleValueIterator */
template<typename _T>
inline
bool
operator==(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() == y.base();
}

//! Return true if the iterators do not have the same base.
/*! \relates SingleValueIterator */
template<typename _T>
inline
bool
operator!=(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return !(x == y);
}

//
// Random access iterator requirements
//

//! Return true if the base of \c x precedes that of \c y.
/*! \relates SingleValueIterator */
template<typename _T>
inline
bool
operator<(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() < y.base();
}

//! Return true if the base of \c x follows that of \c y.
/*! \relates SingleValueIterator */
template<typename _T>
inline
bool
operator>(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() > y.base();
}

//! Return true if the base of \c x precedes or is equal to that of \c y.
/*! \relates SingleValueIterator */
template<typename _T>
inline
bool
operator<=(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() <= y.base();
}

//! Return true if the base of \c x follows or is equal to that of \c y.
template<typename _T>
inline bool
operator>=(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() >= y.base();
}

//! The difference of two iterators.
/*! \relates SingleValueIterator */
template<typename _T>
inline
typename SingleValueIterator<_T>::difference_type
operator-(const SingleValueIterator<_T>& x, const SingleValueIterator<_T>& y)
{
  return x.base() - y.base();
}

//! Iterator advance.
/*! \relates SingleValueIterator */
template<typename _T>
inline
SingleValueIterator<_T>
operator+(typename SingleValueIterator<_T>::difference_type n,
          const SingleValueIterator<_T>& i)
{
  SingleValueIterator<_T> x(i);
  x += n;
  return x;
}

} // namespace ads
}

#endif

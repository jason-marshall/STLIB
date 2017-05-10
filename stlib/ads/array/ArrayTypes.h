// -*- C++ -*-

/*!
  \file ArrayTypes.h
  \brief Defines types for arrays.
*/

#if !defined(__ads_ArrayTypes_h__)
#define __ads_ArrayTypes_h__

#include <boost/call_traits.hpp>

#include <type_traits>

#include <cstddef>

namespace stlib
{
namespace ads
{

//! Defines types for arrays.
template <typename T>
class ArrayTypes
{
public:

  //
  // public types
  //

  //! The element type, \c T, of the array.
  typedef T value_type;
  //! The parameter type.
  /*!
    This is used for passing the value type as an argument.
  */
  typedef typename boost::call_traits<value_type>::param_type parameter_type;
  //! The unqualified value type.
  /*!
    The value type with top level \c const and \c volatile qualifiers removed.
  */
  typedef typename
  std::remove_const<typename std::remove_volatile<value_type>::type>::type
  unqualified_value_type;

  //! A pointer to an array element.
  typedef value_type* pointer;
  //! A pointer to a constant array element.
  typedef const value_type* const_pointer;

  //! An iterator in the array.
  typedef value_type* iterator;
  //! A iterator on constant elements in the array.
  typedef const value_type* const_iterator;

  //! A reference to an array element.
  typedef value_type& reference;
  //! A reference to a constant array element.
  typedef const value_type& const_reference;

  //! The size type is a signed integer.
  /*!
    Having \c std::size_t (which is an unsigned integer) as the size type
    causes minor problems.  Consult "Large Scale C++ Software Design" by
    John Lakos for a discussion of using unsigned integers in a class
    interface.
  */
  typedef int size_type;
  //! Pointer difference type.
  typedef std::ptrdiff_t difference_type;
};

} // namespace ads
} // namespace stlib

#endif

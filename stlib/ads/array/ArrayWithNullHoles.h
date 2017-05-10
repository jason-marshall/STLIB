// -*- C++ -*-

/*!
  \file ArrayWithNullHoles.h
  \brief A class for an N-D sparse array.
*/

#if !defined(__ads_array_ArrayWithNullHoles_h__)
#define __ads_array_ArrayWithNullHoles_h__

#include <boost/call_traits.hpp>

#include <vector>
#include <set>

#include <cassert>

namespace stlib
{
namespace ads
{

//! A 1-D array with holes.
/*!
  \param T is the value type.
*/
template<typename T>
class ArrayWithNullHoles
{
  //
  // Private types.
  //

private:

  typedef std::vector<T> ValueContainer;
  typedef std::vector<std::size_t> IndexContainer;

  //
  // Public types.
  //

public:

  //! The element type of the array.
  typedef T ValueType;
  //! The parameter type.
  /*!
    This is used for passing the value type as an argument.
  */
  typedef typename boost::call_traits<ValueType>::param_type ParameterType;

  //! A pointer to an element.
  typedef typename ValueContainer::pointer Pointer;
  //! A pointer to a constant element.
  typedef typename ValueContainer::const_pointer ConstPointer;

  // CONTINUE: Iterators need to skip the holes.
#if 0
  //! An iterator in the array.
  typedef typename Something Iterator;
  //! A iterator on constant elements in the array.
  typedef typename Something ConstIterator;
#endif

  //! A reference to an array element.
  typedef typename ValueContainer::reference Reference;
  //! A reference to a constant array element.
  typedef typename ValueContainer::const_reference ConstReference;

  //! The size type.
  typedef std::size_t SizeType;
  //! Pointer difference type.
  typedef typename ValueContainer::difference_type DifferenceType;

  //
  // Data.
  //

private:

  //! The array elements.
  ValueContainer _data;
  //! The holes.
  IndexContainer _holes;
  //! The null element.
  ValueType _null;

  //
  // Not implemented.
  //

private:

  //! Default constructor not implemented.
  ArrayWithNullHoles();

public:

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  // @{

  //! Construct from the null value.
  ArrayWithNullHoles(ParameterType null) :
    _data(),
    _holes(),
    _null(null) {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the size of the array (non-null elements and holes combined).
  std::size_t
  size() const
  {
    return _data.size();
  }

  //! Return the number of null elements (holes).
  std::size_t
  sizeNull() const
  {
    return _holes.size();
  }

  //! Return the number of non-null elements.
  std::size_t
  sizeNonNull() const
  {
    return size() - sizeNull();
  }

  //! Return true if the specified element is null.
  bool
  isNull(std::size_t index) const;

  //! Return true if the specified element is non-null.
  bool
  isNonNull(std::size_t index) const;

  //! Return the specified element.
  ParameterType
  get(std::size_t index) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Insert an element into a hole (or at the end if there are no holes).
  /*!
    \return The index of the element.
  */
  std::size_t
  insert(ParameterType value);

  //! Erase the specified element.
  /*!
    \pre The location must not already be a hole.
  */
  void
  erase(std::size_t index);

  //! Erase a range of elements.
  /*!
    \pre The location of each must not already be a hole.
  */
  template<typename IntInputIterator>
  void
  erase(IntInputIterator begin, IntInputIterator end);

  //! Set the specified element.
  /*!
    \pre 0 <= index, index < size(), and value is not null.
  */
  void
  set(std::size_t index, ParameterType value);

  // @}
  //--------------------------------------------------------------------------
  //! \name Validity.
  // @{

  //! Return true if the data structure is valid.
  bool
  isValid() const;

  // @}
};


} // namespace ads
} // namespace stlib

#define __ads_array_ArrayWithNullHoles_ipp__
#include "stlib/ads/array/ArrayWithNullHoles.ipp"
#undef __ads_array_ArrayWithNullHoles_ipp__

#endif

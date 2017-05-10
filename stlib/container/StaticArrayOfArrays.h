// -*- C++ -*-

/*!
  \file StaticArrayOfArrays.h
  \brief A class for a static %array of arrays.
*/

#if !defined(__container_StaticArrayOfArrays_h__)
#define __container_StaticArrayOfArrays_h__

#include "stlib/ext/vector.h"

namespace stlib
{
namespace container
{

//! A static %array of arrays.
/*!
  \param _T is the value type.
*/
template<typename _T>
class StaticArrayOfArrays
{

  //
  // Private types.
  //

private:

  typedef std::vector<_T> Container;

  //
  // Public Types.
  //

public:

  //! The value type.
  typedef typename Container::value_type value_type;
  //! Reference to the value type.
  typedef typename Container::reference reference;
  //! Constant reference to the value type.
  typedef typename Container::const_reference const_reference;
  //! Iterator in the container.
  typedef typename Container::iterator iterator;
  //! Constant iterator in the container.
  typedef typename Container::const_iterator const_iterator;
  //! The size type.
  typedef typename Container::size_type size_type;
  //! The pointer difference type.
  typedef typename Container::difference_type difference_type;
  //! Reverse iterator.
  typedef typename Container::reverse_iterator reverse_iterator;
  //! Constant reverse iterator.
  typedef typename Container::const_reverse_iterator const_reverse_iterator;
  //! A pointer to an %array element.
  typedef typename Container::pointer pointer;
  //! A pointer to a constant %array element.
  typedef typename Container::const_pointer const_pointer;

  //
  // Data.
  //

private:

  //! The %array elements.
  std::vector<value_type> _elements;
  //! Pointers that determine the beginning and end of each %array.
  std::vector<iterator> _pointers;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.  Empty data structure.
  StaticArrayOfArrays() :
    _elements(),
    _pointers(1)
  {
    _pointers[0] = begin();
  }

  //! Construct from a container of containers.
  template<typename _Container>
  StaticArrayOfArrays(const _Container& cOfC);

  //! Construct from the %array sizes and the values.
  template<typename SizeForwardIter, typename ValueForwardIter>
  StaticArrayOfArrays(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd,
                      ValueForwardIter valuesBeginning,
                      ValueForwardIter valuesEnd);

  //! Rebuild from the component %array sizes.
  template<typename SizeForwardIter>
  void
  rebuild(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd);

  //! Rebuild from the %array sizes and the values.
  template<typename SizeForwardIter, typename ValueForwardIter>
  void
  rebuild(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd,
          ValueForwardIter valuesBeginning, ValueForwardIter valuesEnd)
  {
    rebuild(sizesBeginning, sizesEnd);
    std::copy(valuesBeginning, valuesEnd, begin());
  }

  //! Copy constructor.
  StaticArrayOfArrays(const StaticArrayOfArrays& other) :
    _elements(other._elements),
    _pointers(other._pointers.size())
  {
    _pointers[0] = begin();
    for (std::size_t i = 0; i != getNumberOfArrays(); ++i) {
      _pointers[i + 1] = _pointers[i] + other.size(i);
    }
  }

  //! Swap with the argument.
  void
  swap(StaticArrayOfArrays& other)
  {
    _elements.swap(other._elements);
    _pointers.swap(other._pointers);
  }

  //! Assignment operator.
  StaticArrayOfArrays&
  operator=(StaticArrayOfArrays other)
  {
    swap(other);
    return *this;
  }

  // Default destructor is fine.

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors for the whole set of elements.
  // @{

  //! Return the number of arrays.
  size_type
  getNumberOfArrays() const
  {
    return _pointers.size() - 1;
  }

  //! Return the total number of elements.
  size_type
  size() const
  {
    return _elements.size();
  }

  //! Return true if the total number of elements is empty.
  bool
  empty() const
  {
    return _elements.empty();
  }

  //! Return the size of the largest possible %array.
  size_type
  max_size() const
  {
    return _elements.max_size();
  }

  //! Return the memory size.
  size_type
  getMemoryUsage() const
  {
    return sizeof(std::vector<value_type>) +
           _elements.size() * sizeof(value_type) +
           sizeof(std::vector<iterator>) +
           _pointers.size() * sizeof(iterator);
  }

  //! Return a const iterator to the first value.
  const_iterator
  begin() const
  {
    return _elements.begin();
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return _elements.end();
  }

  //! Return a const reverse iterator to the last value.
  const_reverse_iterator
  rbegin() const
  {
    return _elements.rbegin();
  }

  //! Return a const reverse iterator to one before the first value.
  const_reverse_iterator
  rend() const
  {
    return _elements.rend();
  }

  //! Return a const reference to the n<sup>th</sup> overall element.
  const_reference
  operator[](const size_type n) const
  {
    return _elements[n];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors for individual arrays.
  // @{

  //! Return the number of elements in the n<sup>th</sup> %array.
  size_type
  size(const size_type n) const
  {
    return size_type(_pointers[n + 1] - _pointers[n]);
  }

  //! Return true if the n<sup>th</sup> %array is empty.
  bool
  empty(const size_type n) const
  {
    return size(n) == 0;
  }

  //! Return a const iterator to the first value in the n<sup>th</sup> %array.
  const_iterator
  begin(const size_type n) const
  {
    return _pointers[n];
  }

  //! Return a const iterator to one past the last value in the n<sup>th</sup> %array.
  const_iterator
  end(const size_type n) const
  {
    return _pointers[n + 1];
  }

  //! Return a const reverse iterator to the last value in the n<sup>th</sup> %array.
  const_reverse_iterator
  rbegin(const size_type n) const
  {
    return const_reverse_iterator(_pointers[n + 1]);
  }

  //! Return a const reverse iterator to one before the first value in the n<sup>th</sup> %array.
  const_reverse_iterator
  rend(const size_type n) const
  {
    return const_reverse_iterator(_pointers[n]);
  }

  //! Return a const iterator to the first element of the n<sup>th</sup> %array.
  const_iterator
  operator()(const size_type n) const
  {
    return begin(n);
  }

  //! Return the m<sup>th</sup> element of the n<sup>th</sup> %array.
  const_reference
  operator()(const std::size_t n, const std::size_t m) const
  {
#ifdef STLIB_DEBUG
    assert(m < size(n));
#endif
    return *(begin(n) + m);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators for the whole set of elements.
  // @{

  //! Return an iterator to the first value.
  iterator
  begin()
  {
    return _elements.begin();
  }

  //! Return an iterator to one past the last value.
  iterator
  end()
  {
    return _elements.end();
  }

  //! Clear the %array of arrays.
  void
  clear()
  {
    _elements.resize(0);
    _pointers.resize(1);
    _pointers[0] = begin();
  }

  //! Return a reference to the n<sup>th</sup> overall element.
  reference
  operator[](const size_type n)
  {
    return _elements[n];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators for individual arrays.
  // @{

  //! Return an iterator to the first value in the n<sup>th</sup> %array.
  iterator
  begin(const size_type n)
  {
    return _pointers[n];
  }

  //! Return an iterator to one past the last value in the n<sup>th</sup> %array.
  iterator
  end(const size_type n)
  {
    return _pointers[n + 1];
  }

  //! Return an iterator to the first element of the n<sup>th</sup> %array.
  iterator
  operator()(const size_type n)
  {
    return begin(n);
  }

  //! Return the m<sup>th</sup> element of the n<sup>th</sup> %array.
  reference
  operator()(const size_type n, const size_type m)
  {
#ifdef STLIB_DEBUG
    assert(m < size(n));
#endif
    return *(begin(n) + m);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Equality.
  // @{

  //! Return true if the arrays are equal.
  bool
  operator==(const StaticArrayOfArrays& x) const
  {
    if (_elements != x._elements) {
      return false;
    }
    for (size_type n = 0; n != getNumberOfArrays(); ++n) {
      if (size(n) != x.size(n)) {
        return false;
      }
    }
    return true;
  }

  //! Return true if the arrays are not equal.
  bool
  operator!=(const StaticArrayOfArrays& x) const
  {
    return ! operator==(x);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Write to a file stream in ascii format.
  void
  put(std::ostream& out) const;

  //! Read from a file stream in ascii format.
  void
  get(std::istream& in);

  // @}
};

//
// File I/O.
//

//! Write a StaticArrayOfArrays in ascii format.
/*!
  \relates StaticArrayOfArrays

  Below is the file format.
  \verbatim
  number_of_arrays number_of_elements
  array_0_size
  array_0_value_0 array_0_value_1 ...
  array_1_size
  array_1_value_0 array_1_value_1 ...
  ... \endverbatim
*/
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const StaticArrayOfArrays<_T>& x)
{
  x.put(out);
  return out;
}

//! Read a StaticArrayOfArrays in ascii format.
/*!
  \relates StaticArrayOfArrays

  Below is the file format.
  \verbatim
  number_of_arrays number_of_elements
  array_0_size
  array_0_value_0 array_0_value_1 ...
  array_1_size
  array_1_value_0 array_1_value_1 ...
  ... \endverbatim
*/
template<typename _T>
inline
std::istream&
operator>>(std::istream& in, StaticArrayOfArrays<_T>& x)
{
  x.get(in);
  return in;
}

} // namespace container
}

#define __container_StaticArrayOfArrays_ipp__
#include "stlib/container/StaticArrayOfArrays.ipp"
#undef __container_StaticArrayOfArrays_ipp__

#endif

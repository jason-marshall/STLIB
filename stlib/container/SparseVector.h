// -*- C++ -*-

/*!
  \file stlib/container/SparseVector.h
  \brief A class for a sparse vector.
*/

#if !defined(__container_SparseVector_h__)
#define __container_SparseVector_h__

#include "stlib/ads/functor/select.h"
#include "stlib/ads/iterator/TransformIterator.h"
#include "stlib/ext/pair.h"

#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>

#include <cassert>

namespace stlib
{
namespace container
{

USING_STLIB_EXT_PAIR_IO_OPERATORS;

//! A sparse vector.
/*!
  \param _T is the mapped type.

  This is a sorted associative container. The key type is std::size_t. The
  value type is std::pair<key_type, mapped_type>. The interface is similar
  to std::map, however note that the first field in the value type is not
  constant. This is because this class stores the elements in a std::vector.
  (If the first field were constant, the value type would not have a working
  assignment operator.) Note that you are able to modify the keys, but don't
  do that, it will invalidate the data structure.

  The free functions are grouped into the following categories:
  - \ref SparseVectorFile
  - \ref SparseVectorMathematical
  - \ref SparseVectorSparseVector
  - \ref SparseVectorVector
*/
template<typename _T>
class SparseVector
{
  //
  // Public Types.
  //

public:

  //! The key type is std::size_t.
  typedef std::size_t key_type;
  //! The element type is the mapped type.
  typedef _T mapped_type;
  //! The value type is a pair of the constant index and the element value.
  typedef std::pair<key_type, mapped_type> value_type;

  //
  // Private types.
  //

private:

  typedef std::vector<value_type> Container;

  //
  // More public Types.
  //

public:

  //! A pointer to the value type.
  typedef typename Container::pointer pointer;
  //! A pointer to a constant value type.
  typedef typename Container::const_pointer const_pointer;

  //! An iterator to the value type.
  typedef typename Container::iterator iterator;
  //! A const iterator to the value type.
  typedef typename Container::const_iterator const_iterator;

  //! A reverse iterator to the value type.
  typedef typename Container::reverse_iterator reverse_iterator;
  //! A const reverse iterator to the value type.
  typedef typename Container::const_reverse_iterator const_reverse_iterator;

  //! A reference to the value type.
  typedef typename Container::reference reference;
  //! A const reference to the value type.
  typedef typename Container::const_reference const_reference;

  //! The size type.
  typedef typename Container::size_type size_type;
  //! Pointer difference type.
  typedef typename Container::difference_type difference_type;

  //! A const iterator on the keys.
  typedef ads::TransformIterator < const_iterator,
          ads::Select1st<value_type> >
          KeyConstIterator;
  //! A const iterator on the mapped elements.
  typedef ads::TransformIterator < const_iterator,
          ads::Select2nd<value_type> >
          MappedConstIterator;

  //
  // Nested classes.
  //
public:

  //! Compare the indices in the index/value pairs.
  class value_compare :
    public std::binary_function<value_type, value_type, bool>
  {
  public:

    //! Return true if the first index is less than the second.
    bool
    operator()(const value_type& x, const value_type& y) const
    {
      return x.first < y.first;
    }
  };

  //
  // Data.
  //

protected:

  //! The vector of index/element pairs.
  Container _data;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Default constructor.
  SparseVector() :
    _data() {}

  //! Construct a sparse vector from the indices and values.
  SparseVector(const std::vector<value_type>& indexValuePairs) :
    _data(indexValuePairs)
  {
    sort();
  }

  //! Rebuild a sparse vector from the indices and values.
  void
  rebuild(const std::vector<value_type>& indexValuePairs)
  {
    _data = indexValuePairs;
    sort();
  }

  //! Construct a sparse vector from a dense vector of possibly different value type.
  template<typename _T2>
  SparseVector(const std::vector<_T2>& array, const _T2& nullValue);

  //! Construct from the array size (number of non-null elements).
  /*!
    Leave the data uninitialized.
  */
  explicit
  SparseVector(const size_type size) :
    _data(size) {}

  //! Rebuild from the array size (number of non-null elements).
  /*!
    Leave the data uninitialized.
  */
  void
  rebuild(const size_type size)
  {
    _data.resize(size);
  }

  // Default copy constructor and assignment operator are fine.

  //! Swaps data with another SparseVector.
  void
  swap(SparseVector& other)
  {
    _data.swap(other._data);
  }

  //! Copy constructor for different mapped type.
  template<typename _T2>
  SparseVector(const SparseVector<_T2>& other) :
    _data()
  {
    for (typename SparseVector<_T2>::const_iterator i = other.begin();
         i != other.end(); ++i) {
      append(i->first, mapped_type(i->second));
    }
  }

  // The default destructor is fine.

  // @}
  //--------------------------------------------------------------------------
  //! \name Assignment operators.
  // @{

  // Default assignment operator is fine.

  //! Assignment operator for different mapped type.
  template<typename _T2>
  SparseVector&
  operator=(const SparseVector<_T2>& other)
  {
    _data.clear();
    for (typename SparseVector<_T2>::const_iterator i = other.begin();
         i != other.end(); ++i) {
      append(i->first, mapped_type(i->second));
    }
    return *this;
  }

  //! Assignment operator for std::vector. Assume zero is the null value.
  template<typename _T2>
  SparseVector&
  operator=(const std::vector<_T2>& array)
  {
    *this = SparseVector(array, _T2(0));
    return *this;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the number of elements.
  size_type
  size() const
  {
    return _data.size();
  }

  //! Return true if there are no elements.
  bool
  empty() const
  {
    return _data.empty();
  }

  //! Return the value comparison functor.
  value_compare
  value_comp() const
  {
    return value_compare();
  }

  //! Return the possible size.
  size_type
  max_size() const
  {
    return _data.max_size();
  }

  //! Return a const iterator to the beginning of the index/value pairs.
  const_iterator
  begin() const
  {
    return _data.begin();
  }

  //! Return a const iterator to the end of the index/value pairs.
  const_iterator
  end() const
  {
    return _data.end();
  }

  //! Return a const reverse iterator to the beginning of the index/value pairs.
  const_reverse_iterator
  rbegin() const
  {
    return _data.rbegin();
  }

  //! Return a const reverse iterator to the end of the index/value pairs.
  const_reverse_iterator
  rend() const
  {
    return _data.rend();
  }

  //! Return a const pointer to the beginning of the index/value pairs.
  const_pointer
  data() const
  {
    return _data.data();
  }

  //! Return an const iterator to the beginning of the keys.
  KeyConstIterator
  keyBegin() const
  {
    return KeyConstIterator(begin());
  }

  //! Return an const iterator to the end of the keys.
  KeyConstIterator
  keyEnd() const
  {
    return KeyConstIterator(end());
  }

  //! Return an const iterator to the beginning of the mapped values.
  MappedConstIterator
  mappedBegin() const
  {
    return MappedConstIterator(begin());
  }

  //! Return an const iterator to the end of the mapped values.
  MappedConstIterator
  mappedEnd() const
  {
    return MappedConstIterator(end());
  }

  //! Return true if this sparse vector is equal to the argument.
  bool
  operator==(const SparseVector& x) const
  {
    return (_data == x._data);
  }

  //! Return true if this sparse vector is not equal to the argument.
  bool
  operator!=(const SparseVector& x) const
  {
    return ! operator==(x);
  }

  //! Find the element with the specified index.
  const_iterator
  find(const key_type index) const;

  //! Count the number of elements with the specified index.
  /*!
    The return value is either 0 or 1.
  */
  size_type
  count(const key_type index)
  {
    if (find(index) == end()) {
      return 0;
    }
    return 1;
  }

  //! Find the first element whose index is not less than the specified index.
  const_iterator
  lower_bound(const key_type index) const
  {
    return std::lower_bound(_data.begin(), _data.end(),
                            value_type(index, mapped_type()), value_comp());
  }

  //! Find the first element whose index is greater than the specified index.
  const_iterator
  upper_bound(const key_type index) const
  {
    return std::upper_bound(_data.begin(), _data.end(),
                            value_type(index, mapped_type()), value_comp());
  }

  //! Return true if the data structure is valid.
  bool
  isValid() const
  {
    return std::is_sorted(begin(), end(), value_comp());
  }

  // CONTINUE
#if 0
  //! Fill a dense array with the elements from this sparse vector.
  template<typename _T2>
  void
  fill(std::vector<_T2>* array) const;

  //! Fill a dense array using only the non-null elements from this sparse vector.
  template<typename _T2>
  void
  fillNonNull(std::vector<_T2>* array) const;
#endif

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Return an iterator to the beginning of the index/value pairs.
  iterator
  begin()
  {
    return _data.begin();
  }

  //! Return an iterator to the end of the index/value pairs.
  iterator
  end()
  {
    return _data.end();
  }

  //! Return a reverse iterator to the beginning of the index/value pairs.
  reverse_iterator
  rbegin()
  {
    return _data.rbegin();
  }

  //! Return a reverse iterator to the end of the index/value pairs.
  reverse_iterator
  rend()
  {
    return _data.rend();
  }

  //! Return a const pointer to the beginning of the index/value pairs.
  pointer
  data()
  {
    return _data.data();
  }

  //! Find the element with the specified index.
  iterator
  find(const key_type index);

  //! Find the first element whose index is not less than the specified index.
  iterator
  lower_bound(const key_type index)
  {
    return std::lower_bound(_data.begin(), _data.end(),
                            value_type(index, mapped_type()), value_comp());
  }

  //! Find the first element whose index is greater than the specified index.
  iterator
  upper_bound(const key_type index)
  {
    return std::upper_bound(_data.begin(), _data.end(),
                            value_type(index, mapped_type()), value_comp());
  }

  //! Insert the element into the sparse vector.
  /*!
    \return The second part of the pair is true if the element was inserted,
    that is if it was not already present. The first part is an iterator to
    the inserted element.
  */
  std::pair<iterator, bool>
  insert(const value_type& x);

  //! Append the element.
  /*!
    \note the key must be greater than all keys currently in the sparse vector.
  */
  void
  append(const value_type& x)
  {
#ifdef STLIB_DEBUG
    // CONTINUE
    if (!(empty() || _data.back().first < x.first)) {
      std::cerr << "Error in append().\n"
                << *this
                << x.first << ' ' << x.second << '\n';
    }
    assert(empty() || _data.back().first < x.first);
#endif
    _data.push_back(x);
  }

  //! Append the element.
  /*!
    \note the key must be greater than all keys currently in the sparse vector.
  */
  void
  append(const key_type key, const mapped_type& mapped)
  {
    append(value_type(key, mapped));
  }

  //! Erase the pointee.
  void
  erase(iterator pos)
  {
    _data.erase(pos);
  }

  //! Erase the element with the specified index, if it exists.
  /*!
    The return value is the number of elements erased, either 0 or 1.
  */
  size_type
  erase(const key_type index)
  {
    iterator i = find(index);
    if (i == end()) {
      return 0;
    }
    erase(i);
    return 1;
  }

  //! Erase all of the elements.
  void
  clear()
  {
    _data.clear();
  }

  //! Return a reference to the value associated with the index.
  /*!
    Insert an element if necessary.
  */
  mapped_type&
  operator[](const key_type index)
  {
    return insert(value_type(index, mapped_type())).first->second;
  }

  //! Sort the elements by their keys.
  void
  sort()
  {
    std::sort(begin(), end(), value_compare());
  }

  // @}
};

//----------------------------------------------------------------------------
//! \defgroup SparseVectorFile SparseVector File I/O
//@{

//! Write the size and then the space-separated key/value pairs.
/*!
  Format:
  x.size()
  key0 mapped0 key1 mapped1 ...
*/
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const SparseVector<_T>& x)
{
  out << x.size() << '\n';
  for (auto const& element: x) {
    out << element << ' ';
  }
  return out;
}

//! Read the size and then the space-separated key/value pairs.
/*!
  The array will be resized.
*/
template<typename _T>
inline
std::istream&
operator>>(std::istream& in, SparseVector<_T>& x)
{
  typedef typename SparseVector<_T>::iterator iterator;
  std::size_t size;
  in >> size;
  x.rebuild(size);
  for (iterator i = x.begin(); i != x.end(); ++i) {
    in >> *i;
  }
  x.sort();
  return in;
}

//@}
//----------------------------------------------------------------------------
//! \defgroup SparseVectorMathematical SparseVector Mathematical Functions
//@{

//! Return the sum of the components.
template<typename _T>
inline
_T
sum(const SparseVector<_T>& x)
{
  return std::accumulate(x.mappedBegin(), x.mappedEnd(), _T(0));
}

//! Return the product of the components.
template<typename _T>
inline
_T
product(const SparseVector<_T>& x)
{
  return std::accumulate(x.mappedBegin(), x.mappedEnd(), _T(1),
                         std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
template<typename _T>
inline
_T
min(const SparseVector<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.mappedBegin(), x.mappedEnd());
}

//! Return the maximum component.  Use > for comparison.
template<typename _T>
inline
_T
max(const SparseVector<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.mappedBegin(), x.mappedEnd());
}

//@}
//----------------------------------------------------------------------------
//! \defgroup SparseVectorSparseVector Mathematical Operations on Two Sparse Vectors.
//@{

//! Compute the sum of the two arrays.
template<typename _T>
SparseVector<_T>
operator+(const SparseVector<_T>& x, const SparseVector<_T>& y)
{
  return computeBinaryOperation(x, y, std::plus<_T>());
}

//! Compute the difference of the two arrays.
template<typename _T>
SparseVector<_T>
operator-(const SparseVector<_T>& x, const SparseVector<_T>& y)
{
  return computeBinaryOperation(x, y, std::minus<_T>());
}

//! Compute the product of the two arrays.
template<typename _T>
SparseVector<_T>
operator*(const SparseVector<_T>& x, const SparseVector<_T>& y)
{
  return computeBinaryOperation(x, y, std::multiplies<_T>());
}

//! Use the binary function to compute the result.
template<typename _T, typename _BinaryFunction>
SparseVector<_T>
computeBinaryOperation(const SparseVector<_T>& x, const SparseVector<_T>& y,
                       const _BinaryFunction& function);

//@}

} // namespace container
} // namespace stlib

namespace std
{

//----------------------------------------------------------------------------
//! \defgroup SparseVectorVector Operations with vectors and sparse vectors.
//@{

//! += on the non-null elements.
template<typename _T1, typename _T2>
vector<_T1>&
operator+=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y);

//! -= on the non-null elements.
template<typename _T1, typename _T2>
vector<_T1>&
operator-=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y);

//! *= on the non-null elements.
template<typename _T1, typename _T2>
vector<_T1>&
operator*=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y);

//! /= on the non-null elements.
template<typename _T1, typename _T2>
vector<_T1>&
operator/=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y);

//! %= on the non-null elements.
template<typename _T1, typename _T2>
vector<_T1>&
operator%=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y);

//! Perform x += a * y on the non-null elements.
template<typename _T1, typename _T2, typename _T3>
void
scaleAdd(vector<_T1>* x, const _T2 a,
         const stlib::container::SparseVector<_T3>& y);

//@}

} // namespace std


#define __container_SparseVector_ipp__
#include "stlib/container/SparseVector.ipp"
#undef __container_SparseVector_ipp__

#endif

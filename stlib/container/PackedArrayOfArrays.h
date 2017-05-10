// -*- C++ -*-

/*!
  \file PackedArrayOfArrays.h
  \brief A class for a packed %array of arrays.
*/

#if !defined(__container_PackedArrayOfArrays_h__)
#define __container_PackedArrayOfArrays_h__

#include "stlib/ext/vector.h"

#include <boost/config.hpp>

#include <limits>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace container
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;

//! A packed %array of arrays.
/*!
  \param _T is the value type.

  This is an array of arrays that have varying sizes. One may think of it as
  as two dimensional array where the rows have varying numbers of columns.
  This data structure is useful where one might think of using a %container
  of containers like \c std::vector<std::vector<double> >.

  This class packs all of the elements of all of the arrays into a
  contigous storage. This gives it better cache utilization than a
  %container of containers. The downside is that it has limited ability
  to modify the sizes of the component arrays. You can only manipulate
  the last array. We will talk more about that later.
  In addition to packed elements, this class stores a vector of array
  delimiters that define the begining and end of each array.

  There are a number of ways to construct this class. The default
  constructor makes an empty data structure. There are no arrays and
  no elements. Below we construct the class and verify this.
  \code
  container::PackedArrayOfArrays<double> x;
  assert(x.numArrays() == 0);
  assert(x.empty());
  \endcode
  You can also construct it from a %container of containers. You may use
  any %container that satisfies the STL interface for a sequence.
  \code
  std::vector<std::vector<double> > a;
  ...
  container::PackedArrayOfArrays<double> x(a);
  \endcode
  \code
  std::list<std::list<double> > a;
  ...
  container::PackedArrayOfArrays<double> x(a);
  \endcode
  If you know the sizes of the arrays but do not yet know the element values,
  you can construct a PackedArrayOfArrays with the range of sizes.
  \code
  std::vector<std::size_t> sizes;
  ...
  container::PackedArrayOfArrays<double> x(sizes.begin(), sizes.end());
  \endcode
  You can also construct a PackedArrayOfArrays from a range of array sizes
  and the range for the packed elements.
  \code
  std::vector<std::size_t> sizes;
  std::vector<double> packedData;
  ...
  container::PackedArrayOfArrays<double> x(sizes.begin(), sizes.end(),
                                       packedData.begin(), packedData.end());
  \endcode

  This class inherits from std::vector to store the elements, and
  it retains the accessors. Thus you can use \c operator[], \c size(),
  \c empty(), \c begin(), \c end(), \c rbegin(), and \c rend() to work
  with the packed array of elements. For example, below we compute the
  sum of the elements in three different ways.
  \code
  container::PackedArrayOfArrays<double> x;
  ...
  // Indexing.
  double sum = 0;
  for (std::size_t i = 0; i != x.size(); ++i) {
     sum += x[i];
  }
  // Iterator.
  sum = 0;
  for (container::PackedArrayOfArrays<double>::const_iterator i = x.begin();
       i != x.end(); ++i) {
     sum += *i;
  }
  // Accumulate.
  sum = std::accumulate(x.begin(), x.end(), 0.)
  \endcode

  Of course, you can also work with the component arrays. There are
  versions of \c size(), \c empty(), \c begin(), \c end(), \c rbegin(),
  and \c rend() that take the component array index as an argument and
  return information about that array.
  \code
  container::PackedArrayOfArrays<double> x;
  ...
  // Make a vector of the array sizes.
  std::vector<std::size_t> sizes(x.numArrays());
  for (std::size_t i = 0; i != x.numArrays(); ++i) {
     sizes[i] = x.size(i);
  }
  // Calculate the sum of each array.
  std::vector<double> sums(x.numArrays());
  for (std::size_t i = 0; i != x.numArrays(); ++i) {
     sums[i] = std::accumulate(x.begin(i), x.end(i), 0.);
  }
  \endcode

  \c operator() with a single argument will give you the beginning of
  the specified array. This \c x(i) is equivalent to \c x.begin(i).
  With two arguments it gives you the specified element of the
  specified array. That is, \c x(m,n) is the nth element of the mth array.

  Note that in inheriting from std::vector, the modifier functions
  \c assign(), \c insert(), and \c erase() have been disabled.
  This is because (for the sake of efficiency) PackedArrayOfArrays
  only supports changing the size of the last array.
  Use \c push_back() and pop_back() to append an element to the last array
  and remove the last element from the last array, respectively.
  \c pushArray() appends an empty array, while \c popArray()
  erases the last array (which need not be empty).
*/
template<typename _T>
class PackedArrayOfArrays
{
  //
  // Types.
  //
private:

  typedef std::vector<_T> ValueContainer;

public:

  //! The size type.
  typedef typename ValueContainer::size_type size_type;
  //! The value type.
  typedef typename ValueContainer::value_type value_type;
  //! A reference to the value type.
  typedef typename ValueContainer::reference reference;
  //! A const reference to the value type.
  typedef typename ValueContainer::const_reference const_reference;
  //! An iterator over the sequence of values.
  typedef typename ValueContainer::iterator iterator;
  //! A const iterator over the sequence of values.
  typedef typename ValueContainer::const_iterator const_iterator;
  //! A reverse iterator over the sequence of values.
  typedef typename ValueContainer::reverse_iterator reverse_iterator;
  //! A const reverse iterator over the sequence of values.
  typedef typename ValueContainer::const_reverse_iterator
  const_reverse_iterator;

  //
  // Data.
  //
private:

  //! The whole sequence of values for all rows.
  ValueContainer _values;
  //! Delimiters that determine the beginning and end of each %array.
  std::vector<size_type> _delimiters;

  //
  // Disable the following functions from the base class:
  // assign(), insert(), and erase().
  //
private:
  template<typename _InputIterator>
  void
  assign(_InputIterator first, _InputIterator last);

  void
  assign(size_type n, const_reference x);

  iterator
  insert(iterator position, const_reference x);

  void
  insert(iterator position, size_type n,
         const_reference x);

  template<typename _InputIterator>
  void
  insert(iterator position, _InputIterator first,
         _InputIterator last);

  iterator
  erase(iterator position);

  iterator
  erase(iterator first, iterator last);

public:

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  // @{

  //! Default constructor. Empty data structure.
  PackedArrayOfArrays() :
    _values(),
    _delimiters(1)
  {
    _delimiters[0] = 0;
  }

  //! Construct from a %container of containers.
  template<typename _Container>
  PackedArrayOfArrays(const _Container& cOfC);

  //! Construct from the %array sizes and the values.
  template<typename SizeForwardIter, typename ValueForwardIter>
  PackedArrayOfArrays(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd,
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

  //! Swap with the argument.
  void
  swap(PackedArrayOfArrays& other)
  {
    _values.swap(other._values);
    _delimiters.swap(other._delimiters);
  }

  //! Shrink the capacity to match the size.
  void
  shrink_to_fit();

  //! Append the argument.
  void
  append(const PackedArrayOfArrays& other);

  //! Rebuild from the vector of packed arrays.
  /*! Use OpenMP threading if available. */
  void
  rebuild(const std::vector<PackedArrayOfArrays>& parts);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors for the whole set of elements.
  // @{

  //! Return the maximum possible size.
  size_type
  max_size() const BOOST_NOEXCEPT
  {
    return _values.max_size();
  }

  //! Return the capacity of the vector of elements.
  size_type
  capacity() const BOOST_NOEXCEPT
  {
    return _values.capacity();
  }

  //! Return a direct const pointer to the array of values.
  const value_type*
  data() const BOOST_NOEXCEPT
  {
    return _values.data();
  }

  //! Return a const reference to the element in the overall n_th position.
  const_reference
  operator[](const size_type n) const
  {
    return _values[n];
  }

  //! Return the number of arrays.
  size_type
  numArrays() const BOOST_NOEXCEPT
  {
    return _delimiters.size() - 1;
  }

  //! Return the memory usage in bytes.
  size_type
  memoryUsage() const BOOST_NOEXCEPT
  {
    return size() * sizeof(value_type) +
           _delimiters.size() * sizeof(size_type);
  }

  //! Return the memory capacity in bytes.
  size_type
  memoryCapacity() const BOOST_NOEXCEPT
  {
    return _values.capacity() * sizeof(value_type) +
           _delimiters.capacity() * sizeof(size_type);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors for individual arrays.
  // @{

  //! Return the total number of elements.
  size_type
  size() const BOOST_NOEXCEPT
  {
    return _values.size();
  }

  //! Return true if there are no elements.
  bool
  empty() const BOOST_NOEXCEPT
  {
    return _values.empty();
  }

  //! Return a const iterator to the beginning of the sequence of values.
  const_iterator
  begin() const BOOST_NOEXCEPT
  {
    return _values.begin();
  }

  //! Return a const iterator to one past the last element.
  const_iterator
  end() const BOOST_NOEXCEPT
  {
    return _values.end();
  }

  //! Return a const reverse iterator to the last value.
  const_reverse_iterator
  rbegin() const BOOST_NOEXCEPT
  {
    return _values.rbegin();
  }

  //! Return a const reverse iterator to one before the first value.
  const_reverse_iterator
  rend() const BOOST_NOEXCEPT
  {
    return _values.rend();
  }

  //! Return the number of elements in the n<sup>th</sup> %array.
  size_type
  size(const size_type n) const
  {
    return _delimiters[n + 1] - _delimiters[n];
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
    return _values.begin() + _delimiters[n];
  }

  //! Return a const iterator to one past the last value in the n<sup>th</sup> %array.
  const_iterator
  end(const size_type n) const
  {
    return _values.begin() + _delimiters[n + 1];
  }

  //! Return a const reverse iterator to the last value in the n<sup>th</sup> %array.
  const_reverse_iterator
  rbegin(const size_type n) const
  {
    return const_reverse_iterator(end(n));
  }

  //! Return a const reverse iterator to one before the first value in the n<sup>th</sup> %array.
  const_reverse_iterator
  rend(const size_type n) const
  {
    return const_reverse_iterator(begin(n));
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
    return _values[_delimiters[n] + m];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators for the whole set of elements.
  // @{

  //! Return a reference to the element in the overall n_th position.
  reference
  operator[](const size_type n)
  {
    return _values[n];
  }

  //! Return a direct pointer to the array of values.
  value_type*
  data() BOOST_NOEXCEPT
  {
    return _values.data();
  }

  //! Clear the %array of arrays.
  void
  clear()
  {
    _values.clear();
    _delimiters.resize(1);
    _delimiters[0] = 0;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators for individual arrays.
  // @{

  //! Return an iterator to the beginning of the sequence of values.
  iterator
  begin() BOOST_NOEXCEPT
  {
    return _values.begin();
  }

  //! Return an iterator to one past the last element.
  iterator
  end() BOOST_NOEXCEPT
  {
    return _values.end();
  }

  //! Return a reverse iterator to the last value.
  reverse_iterator
  rbegin() BOOST_NOEXCEPT
  {
    return _values.rbegin();
  }

  //! Return a reverse iterator to one before the first value.
  reverse_iterator
  rend() BOOST_NOEXCEPT
  {
    return _values.rend();
  }

  //! Return an iterator to the first value in the n<sup>th</sup> %array.
  iterator
  begin(const size_type n)
  {
    return _values.begin() + _delimiters[n];
  }

  //! Return an iterator to one past the last value in the n<sup>th</sup> %array.
  iterator
  end(const size_type n)
  {
    return _values.begin() + _delimiters[n + 1];
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
    return _values[_delimiters[n] + m];
  }

  //! Append an empty array.
  void
  pushArray()
  {
    _delimiters.push_back(_delimiters.back());
  }

  //! Append an array with elements defined by the sequence.
  template<typename _InputIterator>
  void
  pushArray(_InputIterator begin, _InputIterator end)
  {
    pushArray();
    while (begin != end) {
      push_back(*begin++);
    }
  }

  //! Append \c n empty arrays.
  void
  pushArrays(const std::size_t n)
  {
    _delimiters.insert(_delimiters.end(), n, _delimiters.back());
  }

  //! Pop the last array.
  void
  popArray()
  {
    _values.erase(begin(numArrays() - 1), end(numArrays() - 1));
    _delimiters.pop_back();
  }

  //! Append an element to the last array.
  void
  push_back(const_reference x)
  {
    _values.push_back(x);
    ++_delimiters.back();
  }

  //! Pop an element from the last array.
  void
  pop_back()
  {
    _values.pop_back();
    --_delimiters.back();
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Equality.
  // @{

  //! Return true if the arrays are equal.
  bool
  operator==(const PackedArrayOfArrays& x) const
  {
    return _values == x._values && _delimiters == x._delimiters;
  }

  //! Return true if the arrays are not equal.
  bool
  operator!=(const PackedArrayOfArrays& x) const
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

//! Transpose the sparse array.
/*! \relates PackedArrayOfArrays 

 Determine the number of columns from the values in the input array. 
 Specifically, the number of columns is one more than the maximum value. */
template<typename _Integer>
void
transpose(const PackedArrayOfArrays<_Integer>& input,
          PackedArrayOfArrays<_Integer>* transposed);

//! Transpose the sparse array.
/*! \relates PackedArrayOfArrays

  Use this interface when you want to explicitly specify the number of
  columns in the input array. The values in the input must be less than
  the number of columns. */
template<typename _Integer>
void
transpose(const PackedArrayOfArrays<_Integer>& input, std::size_t numCols,
          PackedArrayOfArrays<_Integer>* transposed);

//! Transpose the sparse array.
/*! \relates PackedArrayOfArrays 

  The element type is a pair of column index and data. For this interface,
  we determine the number of columns from the column index values in the input
  array. Specifically, the number of columns is one more than the maximum 
  value. */
template<typename _Integer, typename _Data>
void
transpose(const PackedArrayOfArrays<std::pair<_Integer, _Data> >& input,
          PackedArrayOfArrays<std::pair<_Integer, _Data> >* transposed);

//! Transpose the sparse array.
/*! \relates PackedArrayOfArrays

  Use this interface when you want to explicitly specify the number of
  columns in the input array. The values in the input must be less than
  the number of columns. */
template<typename _Integer, typename _Data>
void
transpose(const PackedArrayOfArrays<std::pair<_Integer, _Data> >& input,
          const std::size_t numCols,
          PackedArrayOfArrays<std::pair<_Integer, _Data> >* transposed);

//! Add a value to each element.
/*!
  \relates PackedArrayOfArrays
*/
template<typename _T>
inline
PackedArrayOfArrays<_T>&
operator+=(PackedArrayOfArrays<_T>& x, const _T value)
{
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] += value;
  }
  return x;
}

//
// File I/O.
//

//! Write a PackedArrayOfArrays in ascii format.
/*!
  \relates PackedArrayOfArrays

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
operator<<(std::ostream& out, const PackedArrayOfArrays<_T>& x)
{
  x.put(out);
  return out;
}

//! Read a PackedArrayOfArrays in ascii format.
/*!
  \relates PackedArrayOfArrays

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
operator>>(std::istream& in, PackedArrayOfArrays<_T>& x)
{
  x.get(in);
  return in;
}

} // namespace container
}

#define __container_PackedArrayOfArrays_ipp__
#include "stlib/container/PackedArrayOfArrays.ipp"
#undef __container_PackedArrayOfArrays_ipp__

#endif

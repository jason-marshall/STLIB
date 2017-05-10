// -*- C++ -*-

/*!
  \file stlib/container/SimpleMultiArrayConstRef.h
  \brief Multi-dimensional constant %array that references memory and has contiguous storage.
*/

#if !defined(__container_SimpleMultiArrayConstRef_h__)
#define __container_SimpleMultiArrayConstRef_h__

#include "stlib/container/SimpleMultiIndexRange.h"

#include <boost/call_traits.hpp>

namespace stlib
{
namespace container
{

//! Multi-dimensional constant %array that references memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor.
  You can construct an %array from a const pointer to the data and its
  index extents.
  Below we make a 2x4x8 %array with index range [0..1]x[0..3]x[0..7]
  \code
  double data[2 * 4 * 8];
  ...
  container::SimpleMultiArrayConstRef<double, 3>::IndexList extents(2, 4, 8)
  container::SimpleMultiArrayConstRef<double, 3> a(data, extents);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  %array data is referenced.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArrayConstRef<int, 3> b(a);
  \endcode
  The argument may be a SimpleMultiArray, SimpleMultiArrayRef, or a
  SimpleMultiArrayConstRef. The dimension and value type must be the same.
  Since this is a constant %array class, there are no assignment operators.

  You can use rebuild() to make a constant reference to another %array.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArrayConstRef<int, 3> b(a);
  container::SimpleMultiArray<int, 3> c(extents);
  b.rebuild(c);
  \endcode

  <b>Container Member Functions</b>

  SimpleMultiArrayConstRef has the following functionality for treating the
  %array as a constant random access container.

  - empty()
  - size()
  - max_size()
  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()
  - operator()()

  <b>%Array Indexing Member Functions</b>

  SimpleMultiArrayConstRef has the following %array indexing functionality.

  - extents()
  - strides()

  <b>Free Functions</b>

  - \ref SimpleMultiArrayConstRefEquality
  - \ref SimpleMultiArrayConstRefFile
*/
template<typename _T, std::size_t _Dimension>
class SimpleMultiArrayConstRef
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

  // Types for STL compliance.

  //! The size type.
  typedef std::size_t size_type;
  //! Pointer difference type.
  typedef std::ptrdiff_t difference_type;

  //! The element type of the %array.
  typedef _T value_type;
  //! A pointer to a constant %array element.
  typedef const value_type* const_pointer;
  //! A iterator on constant elements in the %array.
  typedef const value_type* const_iterator;
  //! A reverse iterator on constant elements in the %array.
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  //! A reference to a constant %array element.
  typedef const value_type& const_reference;

  // Other types.

  //! An index range.
  typedef SimpleMultiIndexRange<Dimension> Range;
  //! An array index is the same as the size type.
  typedef size_type Index;
  //! A list of indices.
  typedef std::array<Index, Dimension> IndexList;
  //! The parameter type.
  /*! This is used for passing the value type as an argument. */
  typedef typename boost::call_traits<value_type>::param_type Parameter;

  //
  // Member data.
  //
protected:

  //! The %array extents.
  IndexList _extents;
  //! The strides for indexing.
  IndexList _strides;
  //! The number of elements.
  size_type _size;
  //! Pointer to the beginning of a contiguous block of data.
  const_pointer _constData;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor and destructor.
    The default constructor is not implemented.
    The assignment operator is not implemented since one cannot assign to
    const data.
  */
  //@{
public:

  //! Construct from a pointer to the memory and the %array extents.
  SimpleMultiArrayConstRef(const_pointer data, const IndexList& extents) :
    _extents(extents),
    _strides(computeStrides(extents)),
    _size(ext::product(_extents)),
    _constData(data)
  {
  }

  //! Rebuild from a pointer to the memory and the %array extents.
  void
  rebuild(const_pointer data, const IndexList& extents)
  {
    _extents = extents;
    _strides = computeStrides(extents);
    _size = ext::product(_extents);
    _constData = data;
  }

protected:

  //! Compute the strides.
  /*! This is static so it can be called in the initializer list. */
  static
  IndexList
  computeStrides(const IndexList& extents)
  {
    IndexList strides;
    strides[0] = 1;
    for (size_type i = 1; i != Dimension; ++i) {
      strides[i] = strides[i - 1] * extents[i - 1];
    }
    return strides;
  }

private:

  //! Assignment operator not implemented. You cannot assign to const data.
  SimpleMultiArrayConstRef&
  operator=(const SimpleMultiArrayConstRef& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Random access container.
  //@{
public:

  //! Return true if the range is empty.
  bool
  empty() const
  {
    return _size == 0;
  }

  //! Return the size (number of elements) of the range.
  size_type
  size() const
  {
    return _size;
  }

  //! Return the size of the range.
  /*! The max_size and the size are the same. */
  size_type
  max_size() const
  {
    return size();
  }

  //! Return a const iterator to the first value.
  const_iterator
  begin() const
  {
    return data();
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return data() + size();
  }

  //! Return a const reverse iterator to the end of the sequence.
  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  //! Return a const reverse iterator to the beginning of the sequence.
  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  //! Container indexing.
  const_reference
  operator[](const size_type n) const
  {
    return data()[n];
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  //! The range extents.
  const IndexList&
  extents() const
  {
    return _extents;
  }

  //! The index range.
  Range
  range() const
  {
    Range range = {_extents, ext::filled_array<IndexList>(0)};
    return range;
  }

  //! The strides for indexing.
  const IndexList&
  strides() const
  {
    return _strides;
  }

  //! Array indexing.
  const_reference
  operator()(const IndexList& indices) const
  {
#ifdef STLIB_DEBUG
    for (size_type n = 0; n != Dimension; ++n) {
      assert(indices[n] < extents()[n]);
    }
#endif
    return _constData[arrayIndex(indices)];
  }

  //! Array indexing.
  /*! \note The array dimension must be one in order to use this function. */
  const_reference
  operator()(const Index i0) const
  {
#ifdef STLIB_DEBUG
    assert(i0 < extents()[0]);
#endif
    return _constData[arrayIndex(i0)];
  }

  //! Array indexing.
  /*! \note The array dimension must be two in order to use this function. */
  const_reference
  operator()(const Index i0, const Index i1) const
  {
#ifdef STLIB_DEBUG
    assert(i0 < extents()[0] && i1 < extents()[1]);
#endif
    return _constData[arrayIndex(i0, i1)];
  }

  //! Array indexing.
  /*! \note The array dimension must be three in order to use this function. */
  const_reference
  operator()(const Index i0, const Index i1, const Index i2) const
  {
#ifdef STLIB_DEBUG
    assert(i0 < extents()[0] && i1 < extents()[1] && i2 < extents()[2]);
#endif
    return _constData[arrayIndex(i0, i1, i2)];
  }

  //! Return a const pointer to the beginning of the data.
  const_pointer
  data() const
  {
    return _constData;
  }

  //! Return the %array index for the given index list.
  /*!
    This index is in the range [0..size()-1].
  */
  Index
  arrayIndex(const IndexList& indices) const
  {
    return ext::dot(_strides, indices);
  }

  //! Return the %array index for the given list of indices.
  /*!
    This index is in the range [0..size()-1].
    \note The array dimension must be one in order to use this function.
  */
  Index
  arrayIndex(const Index i0) const
  {
    static_assert(Dimension == 1, "Bad dimension.");
    return _strides[0] * i0;
  }

  //! Return the %array index for the given list of indices.
  /*!
    This index is in the range [0..size()-1].
    \note The array dimension must be two in order to use this function.
  */
  Index
  arrayIndex(const Index i0, const Index i1) const
  {
    static_assert(Dimension == 2, "Bad dimension.");
    return _strides[0] * i0 + _strides[1] * i1;
  }

  //! Return the %array index for the given list of indices.
  /*!
    This index is in the range [0..size()-1].
    \note The array dimension must be three in order to use this function.
  */
  Index
  arrayIndex(const Index i0, const Index i1, const Index i2) const
  {
    static_assert(Dimension == 3, "Bad dimension.");
    return _strides[0] * i0 + _strides[1] * i1 + _strides[2] * i2;
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup SimpleMultiArrayConstRefEquality Equality and Comparison Operators
//@{

//! Return true if the arrays have the same extents and elements.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator==(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
           const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return x.extents() == y.extents() &&
         std::equal(x.begin(), x.end(), y.begin());
}

//! Return true if they are not equal.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator!=(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
           const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return !(x == y);
}


//! Lexicographical comparison of the elements.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator<(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
          const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

//! Return y < x.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator>(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
          const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator<=(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
           const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator>=(const SimpleMultiArrayConstRef<_T, _Dimension>& x,
           const SimpleMultiArrayConstRef<_T, _Dimension>& y)
{
  return !(x < y);
}

//@}
//----------------------------------------------------------------------------
//! \defgroup SimpleMultiArrayConstRefFile SimpleMultiArrayConstRef File I/O
//@{

//! Print the %array extents and elements.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out,
           const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
  out << x.extents() << '\n';
  std::copy(x.begin(), x.end(), std::ostream_iterator<_T>(out, "\n"));
  return out;
}

//! Write the %array extents and elements in binary format.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
void
write(std::ostream& out,
      const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
  write(out, x.extents());
  out.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(_T));
}

//! Print the %array by rows with the first row at the bottom.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T>
inline
void
print(std::ostream& out, const SimpleMultiArrayConstRef<_T, 2>& x)
{
  typedef typename SimpleMultiArrayConstRef<_T, 2>::Index Index;
  for (Index j = x.extents()[1] - 1; j >= 0; --j) {
    for (Index i = 0; i != x.extents()[0]; ++i) {
      out << x(i, j) << ' ';
    }
    out << '\n';
  }
}

//@}
//----------------------------------------------------------------------------
/*! \defgroup arraySimpleMultiArrayConstRefMathematical SimpleMultiArrayConstRef Mathematical Functions
*/
//@{

//! Return the sum of the components.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
sum(const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

//! Return the product of the components.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
product(const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
min(const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

//! Return the maximum component.  Use > for comparison.
/*! \relates SimpleMultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
max(const SimpleMultiArrayConstRef<_T, _Dimension>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

//@}

} // namespace container
}

#endif

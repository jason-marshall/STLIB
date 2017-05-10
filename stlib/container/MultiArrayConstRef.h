// -*- C++ -*-

/*!
  \file stlib/container/MultiArrayConstRef.h
  \brief Multi-dimensional constant %array that references memory and has contiguous storage.
*/

#if !defined(__container_MultiArrayConstRef_h__)
#define __container_MultiArrayConstRef_h__

#include "stlib/container/MultiArrayConstView.h"

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
  container::MultiArrayConstRef<double, 3>::SizeList extents(2, 4, 8)
  container::MultiArrayConstRef<double, 3> a(data, extents);
  \endcode

  You can also specify the index bases. Below we make a
  2x4x8 %array with index range [-1..0]x[2..5]x[-3..4]
  \code
  double data[2 * 4 * 8];
  ...
  container::MultiArrayConstRef<double, 3>::SizeList extents(2, 4, 8)
  container::MultiArrayConstRef<double, 3>::IndexList bases(-1, 2, -3)
  container::MultiArrayConstRef<double, 3> a(data, extents, bases);
  \endcode

  The copy constructors create shallow copies of the argument, i.e. the
  %array data is referenced.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArrayConstRef<int, 3> b(a);
  \endcode
  The argument may be a MultiArray, MultiArrayRef, or a MultiArrayConstRef.
  The dimension and value type must be the same.

  Since this is a constant %array class, there are no assignment operators.

  You can use rebuild() to make a constant reference to another %array.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArrayConstRef<int, 3> b(a);
  container::MultiArray<int, 3> c(extents);
  b.rebuild(c);
  \endcode

  <b>Container Member Functions</b>

  MultiArrayConstRef inherits the following functionality for treating the
  %array as a constant random access container.

  - MultiArrayBase::empty()
  - MultiArrayBase::size()
  - MultiArrayBase::max_size()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()

  <b>%Array Indexing Member Functions</b>

  MultiArrayConstRef inherits the following %array indexing functionality.

  - MultiArrayBase::extents()
  - MultiArrayBase::bases()
  - MultiArrayBase::setBases()
  - MultiArrayBase::range()
  - MultiArrayBase::storage()
  - MultiArrayBase::strides()
  - MultiArrayBase::offset()
  - MultiArrayView::operator()()
  - MultiArrayView::view()

  <b>Free Functions</b>

  - \ref MultiArrayConstRefEquality
  - \ref MultiArrayConstRefFile
*/
template<typename _T, std::size_t _Dimension>
class
  MultiArrayConstRef :
  virtual public MultiArrayConstView<_T, _Dimension>
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
private:

  typedef MultiArrayConstView<_T, _Dimension> Base;
  typedef MultiArrayTypes<_T, _Dimension> Types;

public:

  // Types for STL compliance.

  //! The element type of the %array.
  typedef typename Types::value_type value_type;
  //! A pointer to a constant %array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on constant elements in the %array.
  typedef typename Types::const_iterator const_iterator;
  //! A reverse iterator on constant elements in the %array.
  typedef typename Types::const_reverse_iterator const_reverse_iterator;
  //! A reference to a constant %array element.
  typedef typename Types::const_reference const_reference;
  //! The size type.
  typedef typename Types::size_type size_type;
  //! Pointer difference type.
  typedef typename Types::difference_type difference_type;

  // Other types.

  //! The parameter type.
  /*! This is used for passing the value type as an argument. */
  typedef typename Types::Parameter Parameter;
  //! An %array index is a signed integer.
  typedef typename Types::Index Index;
  //! A list of indices.
  typedef typename Types::IndexList IndexList;
  //! A list of sizes.
  typedef typename Types::SizeList SizeList;
  //! The storage order.
  typedef typename Types::Storage Storage;
  //! An index range.
  typedef typename Base::Range Range;
  //! A constant view of this %array.
  typedef typename Base::ConstView ConstView;

  //
  // Using member data.
  //
protected:

  using Base::_constData;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor is fine.

  //! Copy constructor for other data pointer types.
  MultiArrayConstRef(const MultiArrayConstRef<value_type, _Dimension>& other) :
    Base(other)
  {
  }

  //! Construct from a pointer to the memory, the %array extents, and optionally the storage order.
  MultiArrayConstRef(const_pointer data, const SizeList& extents,
                     const Storage& storage = Storage(ColumnMajor())) :
    Base(data, extents, ext::filled_array<IndexList>(0), storage,
         computeStrides(extents, storage))
  {
  }

  //! Construct from a pointer to the memory, the %array extents, the index bases, and optionally the storage order.
  MultiArrayConstRef(const_pointer data, const SizeList& extents,
                     const IndexList& bases,
                     const Storage& storage = Storage(ColumnMajor())) :
    Base(data, extents, bases, storage, computeStrides(extents, storage))
  {
  }

  //! Destructor does not deallocate memory.
  virtual
  ~MultiArrayConstRef()
  {
  }

  //! Copy the data structure. Shallow copy of the elements.
  void
  rebuild(const MultiArrayConstRef& x)
  {
    Base::rebuild(x.data(), x.extents(), x.bases(), x.storage(), x.strides());
  }

  //! Rebuild the data structure.
  /*! \note The size (number of elements) cannot change. */
  void
  rebuild(const SizeList& extents)
  {
    rebuild(extents, bases(), storage());
  }

  //! Rebuild the data structure.
  /*! \note The size (number of elements) cannot change. */
  void
  rebuild(const SizeList& extents, const IndexList& bases)
  {
    rebuild(extents, bases, storage());
  }

  //! Rebuild the data structure.
  /*! \note The size (number of elements) cannot change. */
  void
  rebuild(const SizeList& extents, const IndexList& bases,
          const Storage& storage)
  {
    assert(ext::product(extents) == size());
    rebuild(_constData, extents, bases, storage);
  }

  //! Rebuild the data structure.
  void
  rebuild(const_pointer data, const SizeList& extents, const IndexList& bases,
          const Storage& storage)
  {
    Base::rebuild(data, extents, bases, storage,
                  computeStrides(extents, storage));
  }

protected:

  //! Compute the strides.
  /*!
    This is static so it can be called in the initializer list.
  */
  static
  IndexList
  computeStrides(const SizeList& extents,
                 const Storage& storage = Storage(ColumnMajor()))
  {
    IndexList strides;
    Index s = 1;
    for (size_type i = 0; i != Dimension; ++i) {
      strides[storage[i]] = s;
      s *= extents[storage[i]];
    }
    return strides;
  }

private:

  //! Default constructor not implemented.
  MultiArrayConstRef();

  //! Assignment operator not implemented. You cannot assign to const data.
  MultiArrayConstRef&
  operator=(const MultiArrayConstRef& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Random Access Container.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;

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

  using Base::extents;
  using Base::bases;
  using Base::setBases;
  using Base::range;
  using Base::storage;
  using Base::strides;
  using Base::offset;
  using Base::data;
  using Base::view;
  using Base::arrayIndex;
  using Base::setData;

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup MultiArrayConstRefEquality Equality and Comparison Operators
//@{

//! Return true if the arrays have the same extents and elements.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator==(const MultiArrayConstRef<_T, _Dimension>& x,
           const MultiArrayConstRef<_T, _Dimension>& y)
{
  return x.extents() == y.extents() &&
         std::equal(x.begin(), x.end(), y.begin());
}

//! Return true if they are not equal.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator!=(const MultiArrayConstRef<_T, _Dimension>& x,
           const MultiArrayConstRef<_T, _Dimension>& y)
{
  return !(x == y);
}


//! Lexicographical comparison of the elements.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator<(const MultiArrayConstRef<_T, _Dimension>& x,
          const MultiArrayConstRef<_T, _Dimension>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

//! Return y < x.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator>(const MultiArrayConstRef<_T, _Dimension>& x,
          const MultiArrayConstRef<_T, _Dimension>& y)
{
  return y < x;
}

//! Return !(y < x).
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator<=(const MultiArrayConstRef<_T, _Dimension>& x,
           const MultiArrayConstRef<_T, _Dimension>& y)
{
  return !(y < x);
}

//! Return !(x < y).
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
bool
operator>=(const MultiArrayConstRef<_T, _Dimension>& x,
           const MultiArrayConstRef<_T, _Dimension>& y)
{
  return !(x < y);
}

//@}
//----------------------------------------------------------------------------
//! \defgroup MultiArrayConstRefFile MultiArrayConstRef File I/O
//@{

//! Print the %array extents, index bases, storage, and elements.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out,
           const MultiArrayConstRef<_T, _Dimension>& x)
{
  out << x.extents() << '\n'
      << x.bases() << '\n'
      << x.storage() << '\n';
  for (auto const& element: x) {
    out << element << '\n';
  }
  return out;
}

//! Write the %array extents, index bases, storage, and elements in binary format.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
void
write(const MultiArrayConstRef<_T, _Dimension>& x, std::ostream& out)
{
  stlib::ext::write(x.extents(), out);
  stlib::ext::write(x.bases(), out);
  stlib::ext::write(x.storage(), out);
  out.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(_T));
}

//! Write the %array elements in binary format.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
void
writeElements(const MultiArrayConstRef<_T, _Dimension>& x, std::ostream& out)
{
  out.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(_T));
}

//! Print the %array by rows with the first row at the bottom.
/*! \relates MultiArrayConstRef */
template<typename _T>
inline
void
print(const MultiArrayConstRef<_T, 2>& x, std::ostream& out)
{
  typedef typename MultiArrayConstRef<_T, 2>::Index Index;
  for (Index j = x.bases()[1] + Index(x.extents()[1] - 1); j >= x.bases()[1];
       --j) {
    for (Index i = x.bases()[0]; i != x.bases()[0] + Index(x.extents()[0]);
         ++i) {
      out << x(i, j) << ' ';
    }
    out << '\n';
  }
}

//@}
//----------------------------------------------------------------------------
/*! \defgroup arrayMultiArrayConstRefMathematical MultiArrayConstRef Mathematical Functions

  Note that these functions are defined for
  \ref arrayMultiArrayConstRefMathematical "MultiArrayConstView". We redefine
  them for MultiArrayConstRef because this class has more efficient
  iterators.
*/
//@{

//! Return the sum of the components.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
sum(const MultiArrayConstRef<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

//! Return the product of the components.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
product(const MultiArrayConstRef<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
min(const MultiArrayConstRef<_T, _Dimension>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

//! Return the maximum component.  Use > for comparison.
/*! \relates MultiArrayConstRef */
template<typename _T, std::size_t _Dimension>
inline
_T
max(const MultiArrayConstRef<_T, _Dimension>& x)
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

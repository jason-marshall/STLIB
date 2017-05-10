// -*- C++ -*-

/*!
  \file stlib/container/MultiArray.h
  \brief Multi-dimensional %array that allocates its memory and has contiguous storage.
*/

#if !defined(__container_MultiArray_h__)
#define __container_MultiArray_h__

#include "stlib/container/MultiArrayRef.h"

namespace stlib
{
namespace container
{

//! Multi-dimensional %array that allocates it memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  The default constructor creates an empty %array. Below we make a 3-D %array
  of double precision floating point numbers.
  \code
  container::MultiArray<double, 3> a;
  \endcode

  Below are the declarations of the four constructors that take the %array
  extents as the first argument.
  Each of these constructors takes an optional initial value. If no initial
  value is specified, the %array elements are initialized with the value of
  the default constructor.
  \code
  MultiArray(const SizeList& extents, const value_type& value = value_type());
  MultiArray(const SizeList& extents, const Storage& storage, const value_type& value = value_type());
  MultiArray(const SizeList& extents, const IndexList& bases, const value_type& value = value_type());
  MultiArray(const SizeList& extents, const IndexList& bases, const Storage& storage,
             const value_type& value = value_type());
  \endcode

  Below we make a 2x4x8 %array with index range [0..1]x[0..3]x[0..7] whose
  elements are initialized to 0.
  \code
  container::MultiArray<double, 3>::SizeList extents = {{2, 4, 8}};
  container::MultiArray<double, 3> a(extents);
  \endcode
  Next is an %array with an initial value of 7.
  \code
  container::MultiArray<double, 3> a(extents, 7.);
  \endcode

  Below we make a 2x4x8 %array with index range [-1..0]x[2..5]x[-3..4]
  by specifying the index bases.
  \code
  container::MultiArray<double, 3>::SizeList extents = {{2, 4, 8}};
  container::MultiArray<double, 3>::IndexList bases = {{-1, 2, -3}};
  container::MultiArray<double, 3> a(extents, bases);
  \endcode

  The copy constructors create (deep) copies of the argument.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArray<int, 3> b(a);
  container::MultiArray<double, 3> c = a;
  \endcode
  The argument may be a MultiArray, a MultiArrayRef, or a MultiArrayConstRef .
  The dimension must be the same, but the value type may differ.

  The assignment operators copy the element values. The argument must have
  the same index ranges as the %array, though they can differ in the value
  type.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArray<int, 3> b(extents);
  b = a;
  container::MultiArray<double, 3> c(extents);
  c = a;
  \endcode
  The argument may be any of the multidimensional %array types.

  You can change the shape of an %array with rebuild(). You can specify
  the extents or both the extents and the bases. If the new shape has a
  different number of elements than the old, new memory will be allocated.
  \code
  container::MultiArray<int, 3> a;
  a.rebuild(extents);
  a.rebuild(extents, bases);
  \endcode

  You can also use rebuild() to make a copy of another %array. Again, this
  will allocate new memory if necessary.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArray<int, 3> b;
  b.rebuild(a);
  container::MultiArray<double, 3> c;
  c.rebuild(a);
  \endcode

  <b>Container Member Functions</b>

  MultiArray inherits the following functionality for treating the %array as
  as a random access container.

  - MultiArrayBase::empty()
  - MultiArrayBase::size()
  - MultiArrayBase::max_size()
  - MultiArrayRef::begin()
  - MultiArrayRef::end()
  - MultiArrayRef::rbegin()
  - MultiArrayRef::rend()
  - MultiArrayRef::operator[]()
  - MultiArrayRef::fill()

  <b>%Array Indexing Member Functions</b>

  MultiArray inherits the following %array indexing functionality.

  - MultiArrayBase::extents()
  - MultiArrayBase::bases()
  - MultiArrayBase::setBases()
  - MultiArrayBase::range()
  - MultiArrayBase::storage()
  - MultiArrayBase::strides()
  - MultiArrayBase::offset()
  - MultiArrayView::operator()()
  - MultiArrayView::view()
  - MultiArrayRef::data()

  <b>Free Functions</b>

  - \ref MultiArrayRefAssignmentOperatorsScalar
  - \ref MultiArrayConstRefEquality
  - \ref MultiArrayConstRefFile
  - \ref MultiArrayFile
*/
template<typename _T, std::size_t _Dimension>
class MultiArray : public MultiArrayRef<_T, _Dimension>
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

  typedef MultiArrayTypes<_T, _Dimension> Types;
  typedef MultiArrayRef<_T, _Dimension> Base;
  typedef MultiArrayConstView<_T, _Dimension> VirtualBase;

public:

  // Types for STL compliance.

  //! The element type of the %array.
  typedef typename Types::value_type value_type;
  //! A pointer to an %array element.
  typedef typename Types::pointer pointer;
  //! A pointer to a constant %array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on elements in the %array.
  typedef typename Types::iterator iterator;
  //! A iterator on constant elements in the %array.
  typedef typename Types::const_iterator const_iterator;
  //! A reverse iterator on elements in the %array.
  typedef typename Types::reverse_iterator reverse_iterator;
  //! A reverse iterator on constant elements in the %array.
  typedef typename Types::const_reverse_iterator const_reverse_iterator;
  //! A reference to an %array element.
  typedef typename Types::reference reference;
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
  //! A view of this %array.
  typedef typename Base::View View;

  //
  // Using member data.
  //
protected:

  using Base::_data;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Empty %array.
  MultiArray();

  //! Copy constructor for different types.
  template<typename _T2>
  MultiArray(const MultiArrayConstRef<_T2, _Dimension>& other);

  //! Copy constructor.
  MultiArray(const MultiArray& other);

  //! Construct from the %array extents and optionally an initial value.
  MultiArray(const SizeList& extents, const value_type& value = value_type());

  //! Construct from the %array extents, the storage order, and optionally an initial value.
  MultiArray(const SizeList& extents, const Storage& storage,
             const value_type& value = value_type());

  //! Construct from the %array extents, the index bases, and optionally an initial value.
  MultiArray(const SizeList& extents, const IndexList& bases,
             const value_type& value = value_type());

  //! Construct from the %array extents, the index bases, the storage order, and optionally an initial value.
  MultiArray(const SizeList& extents, const IndexList& bases,
             const Storage& storage, const value_type& value = value_type());

  //! Assignment operator for other %array views.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  MultiArray&
  operator=(const MultiArrayConstView<_T2, _Dimension>& other);

  //! Assignment operator for arrays with contiguous memory.
  /*!
    \pre The arrays must have the same index range.
    \note This version is faster than the assignment operator that takes a
    MultiArrayConstView as an argument because arrays with contiguous memory
    have faster iterators.
  */
  template<typename _T2>
  MultiArray&
  operator=(const MultiArrayConstRef<_T2, _Dimension>& other);

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  MultiArray&
  operator=(const MultiArray& other);

  //! Destructor. Deallocate the memory.
  virtual
  ~MultiArray()
  {
    destroy();
  }

  //! Copy the data structure. Deep copy of the elements.
  template<typename _T2>
  void
  rebuild(const MultiArrayConstRef<_T2, _Dimension>& x)
  {
    // Set the %array shape and allocate memory if necessary.
    rebuild(x.extents(), x.bases(), x.storage());
    // Copy the elements.
    std::copy(x.begin(), x.end(), begin());
  }

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const SizeList& extents)
  {
    rebuild(extents, bases(), storage());
  }

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const SizeList& extents, const IndexList& bases)
  {
    rebuild(extents, bases, storage());
  }

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const SizeList& extents, const IndexList& bases,
          const Storage& storage);

protected:

  using Base::computeStrides;

  void
  destroy()
  {
    if (_data) {
      delete[] _data;
      setData(0);
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Random Access Container.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;
  using Base::begin;
  using Base::end;
  using Base::rbegin;
  using Base::rend;
  using Base::fill;

  //@}
  //--------------------------------------------------------------------------
  //! \name Array Indexing.
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

protected:

  using Base::setData;

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup MultiArrayFile MultiArray File I/O
//@{

//! Read the %array extents, index bases, storage, and elements.
/*!
  \relates MultiArray
  Re-allocate memory if necessary.
*/
template<typename _T, std::size_t _Dimension>
std::istream&
operator>>(std::istream& in, MultiArray<_T, _Dimension>& x);

//! Read the %array extents, index bases, storage, and elements.
/*!
  \relates MultiArray
  Re-allocate memory if necessary.
*/
template<typename _T, std::size_t _Dimension>
void
read(MultiArray<_T, _Dimension>* x, std::istream& in);

//@}

} // namespace container
}

#define __container_MultiArray_ipp__
#include "stlib/container/MultiArray.ipp"
#undef __container_MultiArray_ipp__

#endif

// -*- C++ -*-

/*!
  \file stlib/container/Array.h
  \brief %Array that allocates its memory and has contiguous storage.
*/

#if !defined(__container_Array_h__)
#define __container_Array_h__

#include "stlib/container/ArrayRef.h"

namespace stlib
{
namespace container
{

//! %Array that allocates its memory and has contiguous storage.
/*!
  <b>Constructors, etc.</b>

  The default constructor creates an empty %array. Below we make an %array
  of double precision floating point numbers.
  \code
  container::Array<double> a;
  \endcode

  You can construct an %array from its size. Below we make an
  %array with index range [0..7]
  \code
  container::Array<double> a(8);
  \endcode

  You can also specify the index base. Below we make an
  %array with index range [-3..4]
  \code
  container::Array<double> a(8, -3);
  \endcode

  The copy constructors create (deep) copies of the argument.
  \code
  container::Array<int> a(size);
  container::Array<int> b(a);
  container::Array<double> c = a;
  \endcode
  The argument may be a Array, a ArrayRef, or a ArrayConstRef .
  The value type may differ.

  The assignment operators copy the element values. The argument must have
  the same index ranges as the %array, though they can differ in the value
  type.
  \code
  container::Array<int> a(size);
  container::Array<int> b(size);
  b = a;
  container::Array<double> c(size);
  c = a;
  \endcode
  The argument may be any of the %array types.

  You can change the shape of an %array with rebuild(). You can specify
  the size or both the size and the base. If the new shape has a different
  number of elements than the old, new memory will be allocated.
  \code
  container::Array<int> a;
  a.rebuild(size);
  a.rebuild(size, base);
  \endcode

  You can also use rebuild() to make a copy of another %array. Again, this
  will allocate new memory if necessary.
  \code
  container::Array<int> a(size);
  container::Array<int> b;
  b.rebuild(a);
  container::Array<double> c;
  c.rebuild(a);
  \endcode

  <b>Container Member Functions</b>

  Array inherits the following functionality for treating the %array as
  as a random access container.

  - ArrayBase::empty()
  - ArrayBase::size()
  - ArrayBase::max_size()
  - ArrayRef::begin()
  - ArrayRef::end()
  - ArrayRef::rbegin()
  - ArrayRef::rend()
  - ArrayRef::operator[]()
  - ArrayRef::fill()

  <b>%Array Indexing Member Functions</b>

  Array inherits the following %array indexing functionality.

  - ArrayBase::base()
  - ArrayBase::setBase()
  - ArrayBase::range()
  - ArrayBase::stride()
  - ArrayBase::offset()
  - ArrayView::operator()()
  - ArrayView::view()
  - ArrayRef::data()

  <b>Free Functions</b>

  - \ref ArrayRefAssignmentOperatorsScalar
  - \ref ArrayConstRefEquality
  - \ref ArrayConstRefFile
*/
template<typename _T>
class Array : public ArrayRef<_T>
{
  //
  // Types.
  //
private:

  typedef ArrayTypes<_T> Types;
  typedef ArrayRef<_T> Base;
  typedef ArrayConstView<_T> VirtualBase;

public:

  // Types for STL compliance.

  //! The element type of the array.
  typedef typename Types::value_type value_type;
  //! A pointer to an array element.
  typedef typename Types::pointer pointer;
  //! A pointer to a constant array element.
  typedef typename Types::const_pointer const_pointer;
  //! A iterator on elements in the array.
  typedef typename Types::iterator iterator;
  //! A iterator on constant elements in the array.
  typedef typename Types::const_iterator const_iterator;
  //! A reverse iterator on elements in the array.
  typedef typename Types::reverse_iterator reverse_iterator;
  //! A reverse iterator on constant elements in the array.
  typedef typename Types::const_reverse_iterator const_reverse_iterator;
  //! A reference to an array element.
  typedef typename Types::reference reference;
  //! A reference to a constant array element.
  typedef typename Types::const_reference const_reference;
  //! The size type.
  typedef typename Types::size_type size_type;
  //! Pointer difference type.
  typedef typename Types::difference_type difference_type;

  // Other types.

  //! The parameter type.
  /*! This is used for passing the value type as an argument. */
  typedef typename Types::Parameter Parameter;
  //! An array index is a signed integer.
  typedef typename Types::Index Index;
  //! An index range.
  typedef typename Base::Range Range;
  //! A constant view of this array.
  typedef typename Base::ConstView ConstView;
  //! A view of this array.
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

  //! Default constructor. Empty array.
  Array();

  //! Copy constructor for different types.
  template<typename _T2>
  Array(const ArrayConstRef<_T2>& other);

  //! Copy constructor.
  Array(const Array& other);

  //! Construct from the size.
  Array(size_type size);

  //! Construct from the size and the index base
  Array(size_type size, Index base);

  //! Assignment operator for other array views.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  Array&
  operator=(const ArrayConstView<_T2>& other);

  //! Assignment operator for arrays with contiguous memory.
  /*!
    \pre The arrays must have the same index range.
    \note This version is faster than the assignment operator that takes a
    ArrayConstView as an argument because arrays with contiguous memory
    have faster iterators.
  */
  template<typename _T2>
  Array&
  operator=(const ArrayConstRef<_T2>& other);

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  Array&
  operator=(const Array& other);

  //! Destructor. Deallocate the memory.
  virtual
  ~Array()
  {
    delete[] _data;
    _data = 0;
  }

  //! Copy the data structure. Deepy copy of the elements.
  template<typename _T2>
  void
  rebuild(const ArrayConstRef<_T2>& x)
  {
    // Set the array shape and allocate memory if necessary.
    rebuild(x.size(), x.base());
    // Copy the elements.
    std::copy(x.begin(), x.end(), begin());
  }

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const size_type size)
  {
    rebuild(size, base());
  }

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const size_type size, const Index base);

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

  using Base::base;
  using Base::setBase;
  using Base::range;
  using Base::stride;
  using Base::offset;
  using Base::data;
  using Base::view;

protected:

  using Base::arrayIndex;
  using Base::setData;

  //@}
};

//---------------------------------------------------------------------------
// File I/O.

// CONTINUE: Add input.

} // namespace container
}

#define __container_Array_ipp__
#include "stlib/container/Array.ipp"
#undef __container_Array_ipp__

#endif

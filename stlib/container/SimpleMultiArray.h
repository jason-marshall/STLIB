// -*- C++ -*-

/*!
  \file stlib/container/SimpleMultiArray.h
  \brief Multi-dimensional %array that allocates its memory and has contiguous storage.
*/

#if !defined(__container_SimpleMultiArray_h__)
#define __container_SimpleMultiArray_h__

#include "stlib/container/SimpleMultiArrayRef.h"

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
  container::SimpleMultiArray<double, 3> a;
  \endcode

  You can construct an %array from the index extents and optionally
  an initial value.
  Below we make a 2x4x8 %array with index range [0..1]x[0..3]x[0..7] whose
  elements are initialized to 0.
  \code
  container::SimpleMultiArray<double, 3>::IndexList extents = {{2, 4, 8}};
  container::SimpleMultiArray<double, 3> a(extents);
  \endcode
  Next is an %array with an initial value of 7.
  \code
  container::SimpleMultiArray<double, 3> a(extents, 7.);
  \endcode

  The copy constructors create (deep) copies of the argument.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArray<int, 3> b(a);
  container::SimpleMultiArray<double, 3> c = a;
  \endcode
  The argument may be a SimpleMultiArray, a SimpleMultiArrayRef, or a
  SimpleMultiArrayConstRef . The dimension must be the same, but the value
  type may differ.

  The assignment operators copy the element values. The argument must have
  the same index ranges as the %array, though they can differ in the value
  type. The argument may be any of the multidimensional %array types.
  \code
  container::SimpleMultiArray<int, 3> a(extents);
  container::SimpleMultiArray<int, 3> b(extents);
  b = a;
  container::SimpleMultiArray<double, 3> c(extents);
  c = a;
  \endcode

  You can change the extents of an %array with rebuild(). If the new extents
  have a different number of elements than the old, new memory will be
  allocated.
  \code
  container::SimpleMultiArray<int, 3> a;
  a.rebuild(extents);
  \endcode

  <b>Container Member Functions</b>

  SimpleMultiArray has the following functionality for treating the %array as
  as a random access container.
  - empty()
  - size()
  - max_size()
  - begin()
  - end()
  - rbegin()
  - rend()
  - operator[]()
  - fill()

  <b>%Array Indexing Member Functions</b>

  SimpleMultiArray has the following %array indexing functionality.

  - extents()
  - strides()
  - operator()()
  - data()

  <b>Free Functions</b>

  - \ref SimpleMultiArrayRefAssignmentOperatorsScalar
  - \ref SimpleMultiArrayConstRefEquality
  - \ref SimpleMultiArrayConstRefFile
  - \ref SimpleMultiArrayFile
*/
template<typename _T, std::size_t _Dimension>
class SimpleMultiArray :
  public SimpleMultiArrayRef<_T, _Dimension>
{
  //
  // Types.
  //
private:

  typedef SimpleMultiArrayRef<_T, _Dimension> Base;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Empty %array.
  SimpleMultiArray();

  //! Copy constructor for different types.
  template<typename _T2>
  SimpleMultiArray(const SimpleMultiArrayConstRef<_T2, _Dimension>& other);

  //! Copy constructor.
  SimpleMultiArray(const SimpleMultiArray& other);

  //! Construct from the %array extents.
  SimpleMultiArray(const typename Base::IndexList& extents);

  //! Construct from the %array extents and an initial value.
  SimpleMultiArray(const typename Base::IndexList& extents,
                   const typename Base::value_type& value);

  //! Rebuild the data structure. Re-allocate memory if the size changes.
  void
  rebuild(const typename Base::IndexList& extents);

  //! Assignment operator for arrays with contiguous memory.
  /*! \pre The arrays must have the same index range. */
  template<typename _T2>
  SimpleMultiArray&
  operator=(const SimpleMultiArrayConstRef<_T2, _Dimension>& other);

  //! Assignment operator.
  /*! \pre The arrays must have the same index range. */
  SimpleMultiArray&
  operator=(const SimpleMultiArray& other);

  //! Destructor. Deallocate the memory.
  ~SimpleMultiArray()
  {
    destroy();
  }

  using Base::operator();

protected:

  void
  destroy()
  {
    if (Base::_data) {
      delete[] Base::_data;
      Base::setData(0);
    }
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup SimpleMultiArrayFile SimpleMultiArray File I/O
//@{

//! Read the %array extents, index bases, storage, and elements.
/*!
  \relates SimpleMultiArray
  Re-allocate memory if necessary.
*/
template<typename _T, std::size_t _Dimension>
std::istream&
operator>>(std::istream& in, SimpleMultiArray<_T, _Dimension>& x);

//@}

} // namespace container
}

#define __container_SimpleMultiArray_ipp__
#include "stlib/container/SimpleMultiArray.ipp"
#undef __container_SimpleMultiArray_ipp__

#endif

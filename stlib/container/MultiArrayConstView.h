// -*- C++ -*-

/*!
  \file stlib/container/MultiArrayConstView.h
  \brief Multi-dimensional constant view of an array.
*/

#if !defined(__container_MultiArrayConstView_h__)
#define __container_MultiArrayConstView_h__

#include "stlib/container/MultiArrayTypes.h"
#include "stlib/container/MultiArrayBase.h"
#include "stlib/container/MultiViewIterator.h"

namespace stlib
{
namespace container
{

//! Multi-dimensional constant view of an %array.
/*!
  <b>Constructors, etc.</b>

  Since this %array references externally allocated memory, there is
  no default constructor. This class uses the automatically-generated
  copy constructor; the array data is referenced. You can create
  an instance of this class with the view() member function.

  The copy constructors create shallow copies of the argument, i.e. the
  array data is referenced.
  \code
  container::MultiArray<int, 3> a(extents);
  container::MultiArrayConstView<int, 3> b(a);
  \endcode
  The argument may be any multidimensional %array type, however
  the dimension and value type must be the same.

  <b>Container Member Functions</b>

  MultiArrayConstView inherits the following functionality for treating
  the %array as a random access container.

  - MultiArrayBase::empty()
  - MultiArrayBase::size()
  - MultiArrayBase::max_size()

  It defines the following functions.

  - begin()
  - end()
  - rbegin()
  - rend()

  <b>%Array Indexing Member Functions</b>

  MultiArrayConstView inherits the following %array indexing functionality.

  - MultiArrayBase::extents()
  - MultiArrayBase::bases()
  - MultiArrayBase::setBases()
  - MultiArrayBase::range()
  - MultiArrayBase::storage()
  - MultiArrayBase::strides()
  - MultiArrayBase::offset()

  It defines the following functions.

  - operator()()
  - data()
  - view()

  <b>Free Functions</b>

*/
template<typename _T, std::size_t _Dimension>
class MultiArrayConstView : public MultiArrayBase<_Dimension>
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

  typedef MultiArrayBase<_Dimension> Base;
  typedef MultiArrayTypes<_T, _Dimension> Types;

public:

  // Types for STL compliance.

  //! The element type of the %array.
  typedef typename Types::value_type value_type;
  //! A pointer to a constant %array element.
  typedef typename Types::const_pointer const_pointer;
  //! An iterator on constant elements in the %array.
  typedef MultiViewIterator<MultiArrayConstView, true> const_iterator;
  //! A reverse iterator on constant elements in the %array.
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
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
  typedef MultiArrayConstView<_T, _Dimension> ConstView;

  //
  // Member data.
  //
protected:

  //! Pointer to the beginning of a contiguous block of data.
  const_pointer _constData;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor is fine.

  //! Copy constructor for other data pointer types.
  MultiArrayConstView
  (const MultiArrayConstView<value_type, _Dimension>& other) :
    Base(other),
    _constData(other.data())
  {
  }

  //! Construct from a pointer to the memory, the %array extents, the index bases, the storage order, and the strides.
  MultiArrayConstView(const_pointer data, const SizeList& extents,
                      const IndexList& bases, const Storage& storage,
                      const IndexList& strides) :
    Base(extents, bases, storage, strides),
    _constData(data)
  {
  }

  //! Destructor does not deallocate memory.
  virtual
  ~MultiArrayConstView()
  {
  }

protected:

  //! Rebuild the data structure.
  void
  rebuild(const_pointer data, const SizeList& extents, const IndexList& bases,
          const Storage& storage, const IndexList& strides)
  {
    _constData = data;
    Base::rebuild(extents, bases, storage, strides);
  }

  //! Copy the data structure. Shallow copy of the elements.
  void
  rebuild(const MultiArrayConstView<value_type, _Dimension>& x)
  {
    rebuild(x.data(), x.extents(), x.bases(), x.storage(), x.strides());
  }

private:

  //! Default constructor not implemented.
  /*!
    This class is a virtual base for other classes. Making the default
    constructor private makes sure this class is appropriately constructed.
  */
  MultiArrayConstView();

  //! Assignment operator not implemented. You cannot assign to const data.
  MultiArrayConstView&
  operator=(const MultiArrayConstView& other);

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
    return const_iterator::begin(*this);
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return const_iterator::end(*this);
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

  //! Array indexing.
  const_reference
  operator()(const IndexList& indices) const
  {
#ifdef STLIB_DEBUG
    for (size_type n = 0; n != Dimension; ++n) {
      assert(bases()[n] <= indices[n] &&
             indices[n] < bases()[n] + Index(extents()[n]));
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
    assert(bases()[0] <= i0 && i0 < bases()[0] + Index(extents()[0]));
#endif
    return _constData[arrayIndex(i0)];
  }

  //! Array indexing.
  /*! \note The array dimension must be two in order to use this function. */
  const_reference
  operator()(const Index i0, const Index i1) const
  {
#ifdef STLIB_DEBUG
    assert(bases()[0] <= i0 && i0 < bases()[0] + Index(extents()[0]) &&
           bases()[1] <= i1 && i1 < bases()[1] + Index(extents()[1]));
#endif
    return _constData[arrayIndex(i0, i1)];
  }

  //! Array indexing.
  /*! \note The array dimension must be three in order to use this function. */
  const_reference
  operator()(const Index i0, const Index i1, const Index i2) const
  {
#ifdef STLIB_DEBUG
    assert(bases()[0] <= i0 && i0 < bases()[0] + Index(extents()[0]) &&
           bases()[1] <= i1 && i1 < bases()[1] + Index(extents()[1]) &&
           bases()[2] <= i2 && i2 < bases()[2] + Index(extents()[2]));
#endif
    return _constData[arrayIndex(i0, i1, i2)];
  }

  //! Return a const pointer to the beginning of the data.
  const_pointer
  data() const
  {
    return _constData;
  }

  //! Make a sub-array view with the specified index range.
  /*! The bases for the view are the same as that for the index range. */
  ConstView
  view(const Range& range) const
  {
    // Note: Don't use the function call operator because the range
    // may be empty. In this case array indexing is not valid.
    return ConstView(this->data() + this->arrayIndex(range.bases()),
                     range.extents(), range.bases(), storage(),
                     strides() * range.steps());
  }

  using Base::arrayIndex;

protected:

  //! Set the data pointer.
  void
  setData(const_pointer data)
  {
    _constData = data;
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup arrayMultiArrayConstViewMathematical MultiArrayConstView Mathematical Functions
//@{

//! Return the sum of the components.
/*! \relates MultiArrayConstView */
template<typename _T, std::size_t _Dimension>
inline
_T
sum(const MultiArrayConstView<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

//! Return the product of the components.
/*! \relates MultiArrayConstView */
template<typename _T, std::size_t _Dimension>
inline
_T
product(const MultiArrayConstView<_T, _Dimension>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

//! Return the minimum component.  Use < for comparison.
/*! \relates MultiArrayConstView */
template<typename _T, std::size_t _Dimension>
inline
_T
min(const MultiArrayConstView<_T, _Dimension>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

//! Return the maximum component.  Use > for comparison.
/*! \relates MultiArrayConstView */
template<typename _T, std::size_t _Dimension>
inline
_T
max(const MultiArrayConstView<_T, _Dimension>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

//@}

// CONTINUE: Add equality and file output.

} // namespace container
}

#endif

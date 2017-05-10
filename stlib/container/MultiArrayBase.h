// -*- C++ -*-

/*!
  \file stlib/container/MultiArrayBase.h
  \brief Base class for multi-arrays.
*/

#if !defined(__container_MultiArrayBase_h__)
#define __container_MultiArrayBase_h__

#include "stlib/container/MultiIndexTypes.h"
#include "stlib/container/MultiIndexRange.h"

namespace stlib
{
namespace container
{

//! Base class for multi-arrays.
template<std::size_t _Dimension>
class MultiArrayBase
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

  typedef MultiIndexTypes<_Dimension> Types;

public:

  // Types for STL compliance.

  //! The size type.
  typedef typename Types::size_type size_type;
  //! Pointer difference type.
  typedef typename Types::difference_type difference_type;

  // Other types.

  //! An %array index is a signed integer.
  typedef typename Types::Index Index;
  //! A list of indices.
  typedef typename Types::IndexList IndexList;
  //! A list of sizes.
  typedef typename Types::SizeList SizeList;
  //! The storage order.
  typedef typename Types::Storage Storage;
  //! A multi-index range.
  typedef MultiIndexRange<_Dimension> Range;

  //
  // Member data.
  //
protected:

  //! The %array extents.
  SizeList _extents;
  //! The lower bound for each index.
  IndexList _bases;
  //! The storage order (from least to most significant).
  Storage _storage;
  //! The strides for indexing.
  IndexList _strides;
  //! The offset for indexing the bases.
  Index _offset;
  //! The number of elements.
  size_type _size;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  // The default copy constructor and assignment operator are fine.

  //! Construct from the %array extents, the index bases, the storage order, and the strides.
  MultiArrayBase(const SizeList& extents, const IndexList& bases,
                 const Storage& storage, const IndexList& strides);

  //! Destructor does nothing.
  virtual
  ~MultiArrayBase()
  {
  }

protected:

  //! Rebuild the data structure.
  void
  rebuild(const SizeList& extents, const IndexList& bases,
          const Storage& storage, const IndexList& strides);

private:

  //! Default constructor not implemented.
  MultiArrayBase();

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

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  //! The range extents.
  const SizeList&
  extents() const
  {
    return _extents;
  }

  //! The lower bound for each index.
  const IndexList&
  bases() const
  {
    return _bases;
  }

  //! Set the lower bounds for each index.
  void
  setBases(const IndexList& bases)
  {
    _bases = bases;
    _offset = ext::dot(_strides, _bases);
  }

  //! The index ranges.
  Range
  range() const
  {
    return Range(extents(), bases());
  }

  //! The storage order (from least to most significant).
  const Storage&
  storage() const
  {
    return _storage;
  }

  //! The strides for indexing.
  const IndexList&
  strides() const
  {
    return _strides;
  }

  //! The offset for indexing the bases.
  difference_type
  offset() const
  {
    return _offset;
  }

  //! Return the %array index for the given index list.
  /*!
    For arrays with contiguous storage, this index is in the range
    [0..size()-1].
  */
  Index
  arrayIndex(const IndexList& indices) const
  {
    Index result = 0;
    for (size_type n = 0; n != Dimension; ++n) {
      result += _strides[n] * indices[n];
    }
    return result - _offset;
  }

  //! Return the %array index for the given list of indices.
  /*!
    For arrays with contiguous storage, this index is in the range
    [0..size()-1].
    \note The array dimension must be one in order to use this function.
  */
  Index
  arrayIndex(const Index i0) const
  {
    static_assert(Dimension == 1, "Bad dimension.");
    return _strides[0] * i0 - _offset;
  }

  //! Return the %array index for the given list of indices.
  /*!
    For arrays with contiguous storage, this index is in the range
    [0..size()-1].
    \note The array dimension must be two in order to use this function.
  */
  Index
  arrayIndex(const Index i0, const Index i1) const
  {
    static_assert(Dimension == 2, "Bad dimension.");
    return _strides[0] * i0 + _strides[1] * i1 - _offset;
  }

  //! Return the %array index for the given list of indices.
  /*!
    For arrays with contiguous storage, this index is in the range
    [0..size()-1].
    \note The array dimension must be three in order to use this function.
  */
  Index
  arrayIndex(const Index i0, const Index i1, const Index i2) const
  {
    static_assert(Dimension == 3, "Bad dimension.");
    return _strides[0] * i0 + _strides[1] * i1 + _strides[2] * i2 - _offset;
  }

  //! Calculate the index list for the given array index.
  void
  indexList(const Index n, IndexList* i) const
  {
#ifdef STLIB_DEBUG
    assert(_strides[0] == 1);
#endif
    indexList(n, i, std::integral_constant<std::size_t, _Dimension>());
#ifdef STLIB_DEBUG
    assert(isIn(range(), (*i)));
#endif
  }

private:

  //! Calculate the index list for the given array index.
  void
  indexList(const Index n, IndexList* i,
            std::integral_constant<std::size_t, 1> /*Dimension*/)
  const
  {
    (*i)[0] = n + _bases[0];
  }

  //! Calculate the index list for the given array index.
  void
  indexList(const Index n, IndexList* i,
            std::integral_constant<std::size_t, 2> /*Dimension*/)
  const
  {
    (*i)[1] = n / _strides[1] + _bases[1];
    (*i)[0] = n % _strides[1] + _bases[0];
  }

  //! Calculate the index list for the given array index.
  void
  indexList(Index n, IndexList* i,
            std::integral_constant<std::size_t, 3> /*Dimension*/) const
  {
    (*i)[2] = n / _strides[2] + _bases[2];
    n %= _strides[2];
    (*i)[1] = n / _strides[1] + _bases[1];
    (*i)[0] = n % _strides[1] + _bases[0];
  }

  //@}
};

//----------------------------------------------------------------------------
//! \defgroup MultiArrayBaseEquality Equality Operators
//@{

//! Return true if the member data are equal.
/*! \relates MultiArrayBase */
template<std::size_t _Dimension>
inline
bool
operator==(const MultiArrayBase<_Dimension>& x,
           const MultiArrayBase<_Dimension>& y)
{
  return x.extents() == y.extents() && x.bases() == y.bases() &&
         x.storage() == y.storage() && x.strides() == y.strides();
}

//! Return true if they are not equal.
/*! \relates MultiArrayBase */
template<std::size_t _Dimension>
inline
bool
operator!=(const MultiArrayBase<_Dimension>& x,
           const MultiArrayBase<_Dimension>& y)
{
  return !(x == y);
}

//@}

} // namespace container
}

#define __container_MultiArrayBase_ipp__
#include "stlib/container/MultiArrayBase.ipp"
#undef __container_MultiArrayBase_ipp__

#endif

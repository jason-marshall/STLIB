// -*- C++ -*-

/*!
  \file stlib/container/MultiIndexRange.h
  \brief An index range.
*/

#if !defined(__container_MultiIndexRange_h__)
#define __container_MultiIndexRange_h__

#include "stlib/container/MultiIndexTypes.h"

namespace stlib
{
namespace container
{

USING_STLIB_EXT_ARRAY;

//! An index range.
template<std::size_t _Dimension>
class MultiIndexRange
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

  // Other types.

  //! An %array index is a signed integer.
  typedef typename Types::Index Index;
  //! A list of indices.
  typedef typename Types::IndexList IndexList;
  //! A list of sizes.
  typedef typename Types::SizeList SizeList;

  //
  // Member data.
  //
protected:

  //! The %array extents.
  SizeList _extents;
  //! The lower bound for each index.
  IndexList _bases;
  //! The steps.
  IndexList _steps;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  //@{
public:


  //! Default constructor. Unitilialized data.
  MultiIndexRange()
  {
  }

  //! Construct from the extents, and optionally the bases and the steps.
  MultiIndexRange(const SizeList& extents,
                  const IndexList& bases =
                    ext::filled_array<IndexList>(0),
                  const IndexList& steps =
                    ext::filled_array<IndexList>(1)) :
    _extents(extents),
    _bases(bases),
    _steps(steps)
  {
  }

  //! Construct from the lower and (open) upper bounds.
  MultiIndexRange(const IndexList& lower, const IndexList& upper) :
    _extents(ext::convert_array<size_type>(upper - lower)),
    _bases(lower),
    _steps(ext::filled_array<IndexList>(1))
  {
#ifdef STLIB_DEBUG
    for (std::size_t i = 0; i != Dimension; ++i) {
      assert(lower[i] <= upper[i]);
    }
#endif
  }

#if 0
  //! Construct from the lower and upper bounds, and optionally the steps.
  MultiIndexRange(const IndexList& lower, const IndexList& upper,
                  const IndexList& steps = IndexList(Index(1)));
#endif

  //! Initialize from the extents, and optionally the bases and the steps.
  void
  initialize(const SizeList& extents,
             const IndexList& bases = ext::filled_array<IndexList>(0),
             const IndexList& steps = ext::filled_array<IndexList>(1))
  {
    _extents = extents;
    _bases = bases;
    _steps = steps;
  }

  //! Initialize from the lower and (open) upper bounds.
  void
  initialize(const IndexList& lower, const IndexList& upper)
  {
#ifdef STLIB_DEBUG
    for (std::size_t i = 0; i != Dimension; ++i) {
      assert(lower[i] <= upper[i]);
    }
#endif
    for (std::size_t i = 0; i != Dimension; ++i) {
      _extents[i] = upper[i] - lower[i];
    }
    _bases = lower;
    std::fill(_steps.begin(), _steps.end(), 1);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The index range extents.
  const SizeList&
  extents() const
  {
    return _extents;
  }

  //! The bases.
  const IndexList&
  bases() const
  {
    return _bases;
  }

  //! The (open) upper bound for each index.
  IndexList
  upper() const
  {
    // return _bases + _steps * _extents;
    IndexList upper;
    for (std::size_t i = 0; i != Dimension; ++i) {
      // Convert the extent to a signed integer to avoid implicit
      // conversion to unsigned.
      upper[i] = _bases[i] + _steps[i] * Index(_extents[i]);
    }
    return upper;
  }

  //! The steps.
  const IndexList&
  steps() const
  {
    return _steps;
  }

  //@}
};


//---------------------------------------------------------------------------
// Free functions.

//! Return the intersection of the two index ranges.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension, typename _Size, typename _Index>
MultiIndexRange<_Dimension>
overlap(const std::array<_Size, _Dimension>& extents1,
        const std::array<_Index, _Dimension>& bases1,
        const std::array<_Size, _Dimension>& extents2,
        const std::array<_Index, _Dimension>& bases2);

//! Return the intersection of the two ranges.
/*!
  \pre The ranges must have unit steps.
  \relates MultiIndexRange
*/
template<std::size_t _Dimension>
MultiIndexRange<_Dimension>
overlap(const MultiIndexRange<_Dimension>& x,
        const MultiIndexRange<_Dimension>& y);

//! Return true if the index is in the index range.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
bool
isIn(const MultiIndexRange<_Dimension>& range,
     const typename MultiIndexRange<_Dimension>::IndexList& index);

// CONTINUE: REMOVE
#if 0
//! Return the intersection of the two index ranges.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
inline
MultiIndexRange<_Dimension>
overlap(const MultiArrayBase<_Dimension>& x,
        const MultiArrayBase<_Dimension>& y)
{
  return overlap(x.extents(), x.bases(), y.extents(), y.bases());
}
#endif

//---------------------------------------------------------------------------
// Equality.

//! Return true if the member data are equal.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
inline
bool
operator==(const MultiIndexRange<_Dimension>& x,
           const MultiIndexRange<_Dimension>& y)
{
  return x.extents() == y.extents() && x.bases() == y.bases() &&
         x.steps() == y.steps();
}

//! Return true if they are not equal.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
inline
bool
operator!=(const MultiIndexRange<_Dimension>& x,
           const MultiIndexRange<_Dimension>& y)
{
  return !(x == y);
}

//---------------------------------------------------------------------------
// File I/O.

//! Print the extents, bases, and steps.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out, const MultiIndexRange<_Dimension>& x)
{
  out << x.extents() << '\n'
      << x.bases() << '\n'
      << x.steps() << '\n';
  return out;
}

//! Read the extents, bases, and steps.
/*! \relates MultiIndexRange */
template<std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, MultiIndexRange<_Dimension>& x)
{
  typedef MultiIndexRange<_Dimension> MultiIndexRange;
  typename MultiIndexRange::SizeList extents;
  typename MultiIndexRange::IndexList bases, steps;
  in >> extents >> bases >> steps;
  x = MultiIndexRange(extents, bases, steps);
  return in;
}

} // namespace container
}

#define __container_MultiIndexRange_ipp__
#include "stlib/container/MultiIndexRange.ipp"
#undef __container_MultiIndexRange_ipp__

#endif

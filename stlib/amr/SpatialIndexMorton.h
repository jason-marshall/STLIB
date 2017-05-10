// -*- C++ -*-

/*!
  \file amr/SpatialIndexMorton.h
  \brief N-D spatial index that uses both interlaced and non-interlaced representations.
*/

#if !defined(__amr_SpatialIndexMorton_h__)
#define __amr_SpatialIndexMorton_h__

#include "stlib/amr/MessageInputStream.h"
#include "stlib/amr/MessageOutputStreamChecked.h"
#include "stlib/amr/bits.h"

#include "stlib/numerical/integer/print.h"
#include "stlib/numerical/constants/Logarithm.h"

#include "stlib/ext/array.h"

namespace stlib
{
namespace amr
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! N-D spatial index that uses both interlaced and non-interlaced representations.
/*!
  \param _Dimension The Dimension of the space.
  \param _MaximumLevel The levels are in the range [0 .. _MaximumLevel].

  At the finest level, there are 2<sup>MaximumLevel</sup>
  elements along each dimension.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
class SpatialIndexMorton
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The maximum level.
  BOOST_STATIC_CONSTEXPR std::size_t MaximumLevel = _MaximumLevel;
  //! The number of orthants is 2<sup>Dimension</sup>.
  BOOST_STATIC_CONSTEXPR std::size_t NumberOfOrthants = std::size_t(1) << _Dimension;

  //
  // Public types.
  //
public:

  //! An integer type that can hold the level.
  typedef typename UnsignedInteger
  < numerical::Logarithm < std::size_t, 2, MaximumLevel + 1 >::Result >::Type
  Level;
  //! An integer type that can hold a binary coordinate.
  typedef typename UnsignedInteger<MaximumLevel>::Type Coordinate;
  //! An integer type that can hold the interleaved coordinate code.
  /*!
    The code needs Dimension * MaximumLevel bits to hold the interleaved
    coordinates and one bit to be able to represent a spatial index that
    is larger than all "legitimate" indices. This is useful in algorithms that
    use "past-the-end" iterators.
  */
  typedef typename UnsignedInteger < Dimension* MaximumLevel + 1 >::Type Code;
  //! The vector of Cartesian coordinates.
  typedef std::array<Coordinate, Dimension> CoordinateList;

  //
  // Member data.
  //
private:

  //! The interleaved (left-shifted) coordinates and the validity bit.
  Code _code;
  //! Discrete Cartesian coordinates.
  /*!
    Note that these are right-shifted.  Left shift by
    (MaximumLevel - _level) to get the actual coordinate.
  */
  CoordinateList _coordinates;
  //! The level is in the range [0..MaximumLevel].
  Level _level;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.  Zero level and coordinates.
  SpatialIndexMorton();

  //! Construct from the level and the coordinates.
  SpatialIndexMorton(Level level, const CoordinateList& coordinates);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the interleaved code.
  Code
  getCode() const
  {
    return _code;
  }

  //! Get the level.
  Level
  getLevel() const
  {
    return _level;
  }

  //! Get the coordinates.
  const CoordinateList&
  getCoordinates() const
  {
    return _coordinates;
  }

  //! Return true if this is a valid index.
  /*!
    An invalid index is greater than all valid indices. Specifically, an
    invalid index has a 1 for the (Dimension * MaximumLevel + 1)_th bit.
  */
  bool
  isValid() const
  {
    return !(_code >> Dimension * MaximumLevel);
  }

  //! Return true if the level can be increased.
  bool
  canBeRefined() const
  {
    return _level < MaximumLevel;
  }

  //! Return true if the level can be decreased.
  bool
  canBeCoarsened() const
  {
    return _level > 0;
  }

  //! Get the message stream size for this object.
  static
  std::size_t
  getMessageStreamSize()
  {
    // The invalid flag, the level and the coordinates.
    return sizeof(unsigned char) + sizeof(Level) +
           Dimension * sizeof(Coordinate);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Set the level without updating the code.
  void
  setLevelWithoutUpdating(const Level level)
  {
    _level = level;
  }

  //! Set a coordinate without updating the code.
  void
  setCoordinateWithoutUpdating(const std::size_t dimension,
                               const Coordinate coordinate)
  {
    _coordinates[dimension] = coordinate;
  }

  //! Set the level and coordinates without updating the code.
  void
  setWithoutUpdating(const Level level, const CoordinateList& coordinates)
  {
    _level = level;
    _coordinates = coordinates;
  }

  //! Set the level and coordinates.
  void
  set(Level level, const CoordinateList& coordinates);

  //! Transform to the invalid index.
  void
  invalidate();

  //! Transform to the parent node index.
  /*!
    \pre The level is positive.  (This node has a parent.)
  */
  void
  transformToParent();

  //! Transform to the parent node index.
  /*!
    \pre The level is positive.  (This node has a parent.)
  */
  void
  transformToAncestor(std::size_t steps);

  //! Transform to the specified child node index.
  /*!
    \pre This node must not already be at the deepest level.
  */
  void
  transformToChild(std::size_t n);

  //! Transform to the specified child node index the specified number of times.
  /*!
    \pre The child node must not exceed the deepest level.
  */
  void
  transformToChild(std::size_t n, std::size_t steps);

  //! Transform to the specified neighbor.
  /*!
    \pre This node must have an adjacent neighbor in the specified direction.
  */
  void
  transformToNeighbor(std::size_t n);

  //! Transform to the next node at this level.
  /*!
    \note The last node will be transformed to the first node.
   */
  void
  transformToNext();

private:

  //! Update the code from the coordinates and the level.
  void
  updateCode();

  //! Update the coordinates from the code and the level.
  void
  updateCoordinates();

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  //! Print the level and the coordinates.
  void
  print(std::ostream& out) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Message stream I/O.
  //@{
public:

  //! Write to the message stream.
  void
  write(MessageOutputStream& out) const;

  //! Write to the checked message stream.
  void
  write(MessageOutputStreamChecked& out) const;

  //! Read from the message stream.
  void
  read(MessageInputStream& in);

  //@}
};

//---------------------------------------------------------------------------
// Comparison.

//! Return true if they are equal.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator==(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  // We don't need to check the coordinates.  The code is the interleaved
  // coordinates.
  const bool result = a.getLevel() == b.getLevel() &&
                      a.getCode() == b.getCode();
#ifdef STLIB_DEBUG
  if (result) {
    assert(a.getCoordinates() == b.getCoordinates());
  }
#endif
  return result;
}

//! Equality comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator==
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return a == b.getCode();
}

//! Equality comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator==
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return a.getCode() == b;
}

//! Return true if they are not equal.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator!=(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(a == b);
}

//! Inequality comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator!=
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(a == b);
}

//! Inequality comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator!=
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return !(a == b);
}

//! Less than comparison on the code.
/*!
  \relates SpatialIndexMorton
  \note The code is not sufficient to describe a node.  For example,
  the code 0 can represent the node in the lower corner at any level.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
          const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return a.getCode() < b.getCode();
}

//! Less than comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return a < b.getCode();
}

//! Less than comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return a.getCode() < b;
}

//! Greater than comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
          const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return b < a;
}

//! Greater than comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return b < a;
}

//! Greater than comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return b < a;
}

//! Less than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<=(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(b < a);
}

//! Less than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<=
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(b < a);
}

//! Less than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<=
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return !(b < a);
}

//! Greater than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>=(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(a < b);
}

//! Greater than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>=
(const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code a,
 const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  return !(a < b);
}

//! Greater than or equal to comparison on the code.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator>=
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
 const typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code b)
{
  return !(a < b);
}

//---------------------------------------------------------------------------
// File I/O.

//! Print the spatial index.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
std::ostream&
operator<<(std::ostream& out,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& x)
{
  x.print(out);
  return out;
}

//---------------------------------------------------------------------------
// Topology and geometry.

//! Return true if the key is a local lower corner.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
isLowerCorner(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x);

//! Return true if the key has a parent.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
hasParent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x);

//! Return true if the key has children.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
hasChildren(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x);

//! Return the position of the lower corner of the node.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
computeLocation
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& spatialIndex,
 typename SpatialIndexMorton<_Dimension, _MaximumLevel>::CoordinateList*
 location)
{
  *location = spatialIndex.getCoordinates();
  *location <<= _MaximumLevel - spatialIndex.getLevel();
}

//! Return the position of the lower side of the node.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Coordinate
computeLocation
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& spatialIndex,
 const std::size_t n)
{
  return spatialIndex.getCoordinates()[n] <<
         (_MaximumLevel - spatialIndex.getLevel());
}

//! Compute the length of a side.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
std::size_t
computeLength
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& spatialIndex)
{
  return 1 << (_MaximumLevel - spatialIndex.getLevel());
}

//! Compute the distance between the two nodes.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel>
void
computeSeparations(const SpatialIndexMorton<_Dimension, _MaximumLevel>& index1,
                   const SpatialIndexMorton<_Dimension, _MaximumLevel>& index2,
                   std::array<int, _Dimension>* separations);

//! Return true if the nodes are adjacent.
/*!
  \relates SpatialIndexMorton

  Two nodes in N-D are adjacent if they share a (N-1)-D boundary.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
areAdjacent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
            const SpatialIndexMorton<_Dimension, _MaximumLevel>& b);

//! Return true if the second node is a descendent of the first.
/*!
  \relates SpatialIndexMorton

  \note isDescendent(x, x) is true.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
isDescendent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
             SpatialIndexMorton<_Dimension, _MaximumLevel> descendent);

//! Return true if the second node is an ancestor of the first.
/*!
  \relates SpatialIndexMorton

  \note isAncestor(x, x) is true.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
isAncestor(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
           const SpatialIndexMorton<_Dimension, _MaximumLevel>& ancestor)
{
  return isDescendent(ancestor, node);
}

//! Return true if the node has a neighbor in the specified direction.
/*!
  \relates SpatialIndexMorton
  direction / 2 gives the coordinate.  direction % 2 gives the direction in
  that coordinate (negative or positive).
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
hasNeighbor(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
            std::size_t direction);

//! Return true if there is a next node on same level.
/*!
  \relates SpatialIndexMorton
  The maximum code is
  \[
  (2^{D l} - 1) << ((M - l)D)
  \]
  where \e D is the dimension, \e l is the level, and \e M is the maximum
  level. This function just checks if the code is equal to the maximum value.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
bool
hasNext(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node);

//! Get the adjacent neighbors of the node.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel, typename _OutputIterator>
void
getAdjacentNeighbors(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
                     _OutputIterator adjacent);

//! Get the adjacent neighbors at the next highest level.
/*! \relates SpatialIndexMorton */
template<std::size_t _Dimension, std::size_t _MaximumLevel, typename _OutputIterator>
void
getAdjacentNeighborsHigherLevel
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
 _OutputIterator adjacent);

} // namespace amr
}

#define __amr_SpatialIndexMorton_ipp__
#include "stlib/amr/SpatialIndexMorton.ipp"
#undef __amr_SpatialIndexMorton_ipp__

#endif

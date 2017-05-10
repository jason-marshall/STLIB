// -*- C++ -*-

/*!
  \file geom/spatialIndexing/SpatialIndex.h
  \brief N-D spatial index that uses both interlaced and non-interlaced representations.
*/

#if !defined(__geom_spatialIndexing_SpatialIndex_h__)
#define __geom_spatialIndexing_SpatialIndex_h__

#include "stlib/numerical/integer/bits.h"
#include "stlib/numerical/integer/print.h"
#include "stlib/numerical/constants/Logarithm.h"
#include "stlib/ext/array.h"

namespace stlib
{
namespace geom
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

//! N-D spatial index that uses both interlaced and non-interlaced representations.
/*!
  \param _Dimension The Dimension of the space.
  \param _MaximumLevel The levels are in the range [0 .. _MaximumLevel].

  At the finest level, there are \f$2^{\mathrm{MaximumLevel}}\f$
  elements along each dimension.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
class SpatialIndex
{
  //
  // Enumerations.
  //
public:

  enum {Dimension = _Dimension, MaximumLevel = _MaximumLevel,
        NumberOfOrthants = std::size_t(1) << _Dimension};

  //
  // Public types.
  //
public:

  //! An integer type that can hold the level.
  typedef typename numerical::UnsignedInteger
  < numerical::Logarithm < std::size_t, 2, MaximumLevel + 1 >::Result >::Type
  Level;
  //! An integer type that can hold a binary coordinate.
  typedef typename numerical::UnsignedInteger<MaximumLevel>::Type
  Coordinate;
  //! An integer type that can hold the interleaved coordinate code.
  typedef typename numerical::UnsignedInteger<Dimension* MaximumLevel>::Type
  Code;

  //
  // Member data.
  //
private:

  //! The interleaved (left-shifted) coordinates.
  Code _code;
  //! Discrete Cartesian coordinates.
  /*!
    Note that these are right-shifted.  Left shift by
    (MaximumLevel - _level) to get the actual coordinate.
  */
  std::array<Coordinate, Dimension> _coordinates;
  //! The level is in the range [0..MaximumLevel].
  Level _level;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.  Zero level and coordinates.
  SpatialIndex() :
    _code(0),
    _coordinates(),
    _level(0)
  {
    std::fill(_coordinates.begin(), _coordinates.end(), Coordinate(0));
    static_assert(Dimension > 0, "Bad dimension.");
  }

  //! Construct from the level and the coordinates.
  SpatialIndex(const Level level,
               const std::array<Coordinate, Dimension>& coordinates) :
    _code(),
    _coordinates(coordinates),
    _level(level)
  {
    static_assert(Dimension > 0, "Bad dimension.");
    updateCode();
  }

  //! Copy constructor.
  SpatialIndex(const SpatialIndex& other) :
    _code(other._code),
    _coordinates(other._coordinates),
    _level(other._level)
  {
  }

  //! Assignment operator.
  SpatialIndex&
  operator=(const SpatialIndex& other)
  {
    if (this != &other) {
      _code = other._code;
      _coordinates = other._coordinates;
      _level = other._level;
    }
    return *this;
  }

  //! Destructor.
  ~SpatialIndex()
  {
  }

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
  const std::array<Coordinate, Dimension>&
  getCoordinates() const
  {
    return _coordinates;
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

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Set the level and coordinates.
  void
  set(const Level level,
      const std::array<Coordinate, Dimension> coordinates)
  {
    _level = level;
    _coordinates = coordinates;
    updateCode();
  }

  //! Transform to the parent node index.
  /*!
    \pre The level is positive.  (This node has a parent.)
  */
  void
  transformToParent()
  {
#ifdef STLIB_DEBUG
    assert(_level != 0);
#endif
    --_level;
    _coordinates >>= 1;
    updateCode();
  }

  //! Transform to the specified child node index.
  /*!
    \pre This node must not already be at the deepest level.
  */
  void
  transformToChild(std::size_t n)
  {
#ifdef STLIB_DEBUG
    assert(n < NumberOfOrthants);
#endif
    ++_level;
    _coordinates <<= 1;
    for (std::size_t i = 0; i != Dimension; ++i) {
      _coordinates[i] += n % 2;
      n >>= 1;
    }
    updateCode();
  }

  //! Transform to the specified child node index the specified number of times.
  /*!
    \pre The child node must not exceed the deepest level.
  */
  void
  transformToChild(const std::size_t n, const std::size_t steps)
  {
#ifdef STLIB_DEBUG
    assert(n < NumberOfOrthants);
#endif
    _level += steps;
    for (std::size_t s = 0; s != steps; ++s) {
      _coordinates <<= 1;
      std::size_t m = n;
      for (std::size_t i = 0; i != Dimension; ++i) {
        _coordinates[i] += m % 2;
        m >>= 1;
      }
    }
    updateCode();
  }

  //! Transform to the specified neighbor.
  /*!
    \pre This node must have an adjacent neighbor in the specified direction.
  */
  void
  transformToNeighbor(std::size_t n);

private:

  void
  updateCode()
  {
    // Make a copy of the coordinates.  We will right-shift them to strip
    // off the binary digits.
    std::array<Coordinate, Dimension> c(_coordinates);
    // First left-shift to get the real coordinates.
    c <<= (MaximumLevel - _level);
    // Interlace the coordinates.
    _code = numerical::interlaceBits<Code>(c, MaximumLevel);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  //! Print the level and the coordinates.
  void
  print(std::ostream& out) const
  {
    out << double(_level) << " ";
    for (std::size_t i = 0; i != Dimension; ++i) {
      numerical::printBits(out, _coordinates[i]);
      out << " ";
    }
    numerical::printBits(out, _code);
  }

  //@}
};

//---------------------------------------------------------------------------
// Equality.

//! Return true if they are equal.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator==(const SpatialIndex<_Dimension, _MaximumLevel>& a,
           const SpatialIndex<_Dimension, _MaximumLevel>& b)
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

//! Return true if they are not equal.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator!=(const SpatialIndex<_Dimension, _MaximumLevel>& a,
           const SpatialIndex<_Dimension, _MaximumLevel>& b)
{
  return !(a == b);
}


//! Less than comparison on the code.
/*!
  \relates SpatialIndex
  \note The code is not sufficient to describe a node.  For example,
  the code 0 can represent the node in the lower corner at any level.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
operator<(const SpatialIndex<_Dimension, _MaximumLevel>& a,
          const SpatialIndex<_Dimension, _MaximumLevel>& b)
{
  return a.getCode() < b.getCode();
}

//---------------------------------------------------------------------------
// File I/O.

//! Print the spatial index.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
std::ostream&
operator<<(std::ostream& out,
           const SpatialIndex<_Dimension, _MaximumLevel>& x)
{
  x.print(out);
  return out;
}

//---------------------------------------------------------------------------
// Topology and geometry.

//! Return true if the key is a local lower corner.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
isLowerCorner(const SpatialIndex<_Dimension, _MaximumLevel>& x)
{
  std::size_t sum = 0;
  for (std::size_t d = 0; d != _Dimension; ++d) {
    sum += x.getCoordinates()[d] % 2;
  }
  return sum == 0;
}


//! Return true if the key has a parent.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasParent(const SpatialIndex<_Dimension, _MaximumLevel>& x)
{
  return x.getLevel() != 0;
}

//! Return the position of the lower corner of the node.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
computeLocation(const SpatialIndex<_Dimension, _MaximumLevel>& spatialIndex,
                std::array <
                typename SpatialIndex<_Dimension, _MaximumLevel>::Coordinate,
                _Dimension > *
                location)
{
  *location = spatialIndex.getCoordinates();
  *location <<= _MaximumLevel - spatialIndex.getLevel();
}

//! Return the position of the lower side of the node.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
typename SpatialIndex<_Dimension, _MaximumLevel>::Coordinate
computeLocation(const SpatialIndex<_Dimension, _MaximumLevel>& spatialIndex,
                const std::size_t n)
{
  return spatialIndex.getCoordinates()[n] <<
         (_MaximumLevel - spatialIndex.getLevel());
}

//! Compute the length of a side.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
std::size_t
computeLength(const SpatialIndex<_Dimension, _MaximumLevel>& spatialIndex)
{
  return 1 << (_MaximumLevel - spatialIndex.getLevel());
}

//! Compute the distance between the two nodes.
/*!
  \relates SpatialIndex
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
computeSeparations(const SpatialIndex<_Dimension, _MaximumLevel>& index1,
                   const SpatialIndex<_Dimension, _MaximumLevel>& index2,
                   std::array<int, _Dimension>* separations)
{
  typedef SpatialIndex<_Dimension, _MaximumLevel> SpatialIndex;
  typedef typename SpatialIndex::Coordinate Coordinate;

  std::array<Coordinate, _Dimension> location1, location2;
  computeLocation(index1, &location1);
  computeLocation(index2, &location2);
  const std::size_t length1 = computeLength(index1);
  const std::size_t length2 = computeLength(index2);
  for (std::size_t d = 0; d != _Dimension; ++d) {
    (*separations)[d] = (location1[d] < location2[d] ?
                         (location2[d] - location1[d]) - length1 :
                         (location1[d] - location2[d]) - length2);
  }
}

//! Return true if the nodes are adjacent.
/*!
  \relates SpatialIndex

  Two nodes in N-D are adjacent if they share a (N-1)-D boundary.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
areAdjacent(const SpatialIndex<_Dimension, _MaximumLevel>& a,
            const SpatialIndex<_Dimension, _MaximumLevel>& b)
{
  std::array<int, _Dimension> separations;
  computeSeparations(a, b, &separations);
  std::size_t countZero = 0, countNegative = 0;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    countZero += separations[i] == 0;
    countNegative += separations[i] < 0;
  }
  return countZero == 1 && countNegative == _Dimension - 1;
}

//! Return true if the node has a neighbor in the specified direction.
/*!
  \relates SpatialIndex
  direction / 2 gives the coordinate.  direction % 2 gives the direction in
  that coordinate (negative or positive).
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasNeighbor(const SpatialIndex<_Dimension, _MaximumLevel>& node,
            const int direction)
{
  typedef typename SpatialIndex<_Dimension, _MaximumLevel>::Coordinate
  Coordinate;

  // Negative direction.
  if (direction % 2 == 0) {
    // The lower side of the node.
    return node.getCoordinates()[direction / 2] != 0;
  }
  // Positive direction.
  else {
    // The upper side of the node.
    Coordinate upper = computeLocation(node, direction / 2)
                       + computeLength(node);
    // The coordinate may have more than _MaximumLevel digits.  Keep only
    // _MaximumLevel digits.
    upper <<= std::numeric_limits<Coordinate>::digits - _MaximumLevel;
    return upper != 0;
  }
}

} // namespace geom
}

#define __geom_spatialIndexing_SpatialIndex_ipp__
#include "stlib/geom/spatialIndexing/SpatialIndex.ipp"
#undef __geom_spatialIndexing_SpatialIndex_ipp__

#endif

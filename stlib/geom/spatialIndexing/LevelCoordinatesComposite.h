// -*- C++ -*-

/*!
  \file geom/spatialIndexing/LevelCoordinatesComposite.h
  \brief CONTINUE
*/

#if !defined(__geom_spatialIndexing_LevelCoordinatesComposite_h__)
#define __geom_spatialIndexing_LevelCoordinatesComposite_h__

#include "stlib/numerical/integer/bits.h"
#include "stlib/numerical/integer/print.h"
#include "stlib/numerical/constants/Logarithm.h"

namespace stlib
{
namespace geom
{

//! The level and coordinates as a composite number.
template<std::size_t _Dimensions, std::size_t NumLev>
class
  LevelCoordinatesComposite;


// CONTINUE: Consider a base class.
//! The level and coordinates as a composite number.
template<std::size_t NumLev>
class LevelCoordinatesComposite<2, NumLev>
{
  //
  // Enumerations.
  //
public:

  enum {Dimensions = 2, NumberOfLevels = NumLev};

  //
  // Public types.
  //
public:

  //! An integer type that can hold the level.
  typedef typename numerical::UnsignedInteger
  <numerical::Logarithm<std::size_t, 2, NumberOfLevels>::Result>::Type Level;
  //! An integer type that can hold a binary coordinate.
  typedef typename numerical::UnsignedInteger < NumberOfLevels - 1 >::Type
  Coordinate;

  //
  // Member data.
  //
private:

  Level _level;
  Coordinate _coordinate0, _coordinate1;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.  Zero level and coordinates.
  LevelCoordinatesComposite() :
    _level(),
    _coordinate0(),
    _coordinate1()
  {
  }

  //! Construct from the level and the coordinates.
  LevelCoordinatesComposite(const Level level, const Coordinate coordinate0,
                            const Coordinate coordinate1) :
    _level(level),
    _coordinate0(coordinate0),
    _coordinate1(coordinate1)
  {
  }

  //! Copy constructor.
  LevelCoordinatesComposite(const LevelCoordinatesComposite& other) :
    _level(other._level),
    _coordinate0(other._coordinate0),
    _coordinate1(other._coordinate1)
  {
  }

  //! Assignment operator.
  LevelCoordinatesComposite&
  operator=(const LevelCoordinatesComposite& other)
  {
    if (this != &other) {
      _level = other._level;
      _coordinate0 = other._coordinate0;
      _coordinate1 = other._coordinate1;
    }
    return *this;
  }

  //! Destructor.
  ~LevelCoordinatesComposite()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the level.
  Level
  getLevel() const
  {
    return _level;
  }

  //! Get the first coordinate.
  Coordinate
  getCoordinate0() const
  {
    return _coordinate0;
  }

  //! Get the second coordinate.
  Coordinate
  getCoordinate1() const
  {
    return _coordinate1;
  }

  //! Return true if the level can be increased.
  bool
  canBeRefined() const
  {
    return _level < NumberOfLevels - 1;
  }

  //! Return true if the level can be decreased.
  bool
  canBeCoarsened() const
  {
    return _level > 0;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  bool
  operator==(const LevelCoordinatesComposite& other) const
  {
    return _level == other._level &&
           _coordinate0 == other._coordinate0 &&
           _coordinate1 == other._coordinate1;
  }

  bool
  operator<(const LevelCoordinatesComposite& other) const
  {
    return _level < other._level ||
           (_level == other._level &&
            _coordinate0 < other._coordinate0) ||
           (_level == other._level &&
            _coordinate0 == other._coordinate0 &&
            _coordinate1 < other._coordinate1);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Set the level.
  void
  setLevel(const Level level)
  {
    _level = level;
  }

  //! Set the first coordinate.
  void
  setCoordinate0(const Coordinate coordinate)
  {
    _coordinate0 = coordinate;
  }

  //! Set the second coordinate.
  void
  setCoordinate1(const Coordinate coordinate)
  {
    _coordinate1 = coordinate;
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
    _coordinate0 >>= 1;
    _coordinate1 >>= 1;
  }

  //! Transform to the specified child node index.
  /*!
    \pre This node must not already be at the deepest level.
  */
  void
  transformToChild(const std::size_t n)
  {
#ifdef STLIB_DEBUG
    assert(0 <= n && n < 4);
#endif
    transformToChild(n % 2, n / 2);
  }

  //! Transform to the specified child node index.
  /*!
    \pre This node must not already be at the deepest level.
  */
  void
  transformToChild(const bool x, const bool y)
  {
#ifdef STLIB_DEBUG
    assert(_level != NumberOfLevels - 1);
#endif
    ++_level;
    _coordinate0 <<= 1;
    _coordinate0 += x;
    _coordinate1 <<= 1;
    _coordinate1 += y;
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
    numerical::printBits(out, _coordinate0);
    out << " ";
    numerical::printBits(out, _coordinate1);
  }

  //@}
};


//! Print the spatial index.
/*!
  \relates LevelCoordinatesComposite
*/
template<std::size_t _Dimensions, std::size_t NumLev>
inline
std::ostream&
operator<<(std::ostream& out,
           const LevelCoordinatesComposite<_Dimensions, NumLev>& x)
{
  x.print(out);
  return out;
}

//! Return true if the key is a local lower corner.
/*!
  \relates LevelCoordinatesComposite
*/
template<std::size_t NumLev>
inline
bool
isLowerCorner(const LevelCoordinatesComposite<2, NumLev>& x)
{
  return x.getCoordinate0() % 2 == 0 && x.getCoordinate1() % 2 == 0;
}


//! Return true if the key has a parent.
/*!
  \relates LevelCoordinatesComposite
*/
template<std::size_t _Dimensions, std::size_t NumLev>
inline
bool
hasParent(const LevelCoordinatesComposite<_Dimensions, NumLev>& x)
{
  return x.getLevel() != 0;
}


} // namespace geom
}

#define __geom_spatialIndexing_LevelCoordinatesComposite_ipp__
#include "stlib/geom/spatialIndexing/LevelCoordinatesComposite.ipp"
#undef __geom_spatialIndexing_LevelCoordinatesComposite_ipp__

#endif

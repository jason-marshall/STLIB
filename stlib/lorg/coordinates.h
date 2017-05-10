// -*- C++ -*-

/*!
  \file lorg/coordinates.h
  \brief Convert N-D floating-point coordinates to discrete coordinates.
*/

#if !defined(__lorg_coordinates_h__)
#define __lorg_coordinates_h__

#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace lorg
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Convert N-D floating-point coordinates to discrete coordinates.
/*!
  \param _Integer The unsigned integer type to use for coordinates.
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.
*/
template<typename _Integer, typename _Float, std::size_t _Dimension>
class DiscreteCoordinates
{
  //
  // Types.
  //
public:
  //! A cartesian point.
  typedef std::array<_Float, _Dimension> Point;
  //! A point with integer coordinates.
  typedef std::array<_Integer, _Dimension> DiscretePoint;

  //
  // Constants.
  //
public:
  //! The number of levels of refinement.
  BOOST_STATIC_CONSTEXPR std::size_t NumLevels =
    std::numeric_limits<_Integer>::digits / _Dimension;

  //
  // Member data.
  //
private:
  //! The lower corner of the Cartesian domain.
  Point _lowerCorner;
  //! The scaling factor in transforming to cells is the extent divided by the domain length.
  _Float _scaling;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Construct from the positions and (optionally) the feature length.
  DiscreteCoordinates(const std::vector<Point>& positions);

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Calculate the discrete coordinates.
  DiscretePoint
  discretize(const Point& p) const;

  //@}
};

} // namespace lorg
}

#define __lorg_coordinates_tcc__
#include "stlib/lorg/coordinates.tcc"
#undef __lorg_coordinates_tcc__

#endif

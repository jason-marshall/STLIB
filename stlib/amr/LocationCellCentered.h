// -*- C++ -*-

/*!
  \file amr/LocationCellCentered.h
  \brief Computes the location of cell centers in arrays.
*/

#if !defined(__amr_LocationCellCentered_h__)
#define __amr_LocationCellCentered_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace amr
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Computes the location of cell centers in arrays.
/*!
  \param _Traits Traits for the orthtree.
*/
template<typename _Traits>
class LocationCellCentered
{
  //
  // Public types.
  //
public:

  //! The number type.
  typedef typename _Traits::Number Number;
  //! A Cartesian position.
  typedef std::array<Number, _Traits::Dimension> Point;

  //
  // Member data.
  //
private:

  Point _offsets, _origin;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   Use the synthesized copy constructor, assignment operator, and
   destructor. */
  //@{
public:

  //! Construct from the Cartesian lower corner, the Cartesian extents and the array extents.
  template<typename _ExtentList>
  LocationCellCentered(const Point& lowerCorner, const Point& cartesianExtents,
                       const _ExtentList& arrayExtents) :
    _offsets(cartesianExtents / ext::convert_array<Number>(arrayExtents)),
    _origin(lowerCorner + 0.5 * _offsets)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Compute the location for the specified cell center.
  template<typename _IndexList>
  void
  operator()(const _IndexList& index, Point* location) const
  {
    for (std::size_t n = 0; n != _Traits::Dimension; ++n) {
      (*location)[n] = _origin[n] + index[n] * _offsets[n];
    }
  }

  //! Compute the location for the specified cell center.
  template<typename _IndexList>
  Point
  operator()(const _IndexList& index) const
  {
    Point location;
    for (std::size_t n = 0; n != _Traits::Dimension; ++n) {
      location[n] = _origin[n] + index[n] * _offsets[n];
    }
    return location;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if the data structures are equal.
  bool
  operator==(const LocationCellCentered& other)
  {
    return _offsets == other._offsets && _origin == other._origin;
  }

  //@}
};

} // namespace amr
}

#endif

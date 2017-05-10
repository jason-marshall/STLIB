// -*- C++ -*-

#if !defined(__geom_ParametrizedLine_h__)
#define __geom_ParametrizedLine_h__

#include <iostream>
#include <cassert>
#include <cmath>

#include "stlib/geom/kernel/SegmentMath.h"

namespace stlib
{
namespace geom
{

//! A parametrized line in N-D.
/*!
  \param N is the space dimension.
  \param T is the number type. By default it is double.
 */
template < std::size_t N, typename T = double >
class ParametrizedLine
{
private:

  //
  // Private types.
  //

  //! The segment upon which this line is built.
  typedef SegmentMath<N, T> Segment;

public:

  //
  // Public types.
  //

  //! The number type.
  typedef T Number;
  //! The Cartesian point type.
  typedef std::array<Number, N> Point;
  //! The parameter point type.
  typedef std::array<Number, 1> ParameterPoint;

private:

  //
  // Member data
  //

  // The line segment that determines the line and the parametrization.
  Segment _segment;
  // The derivative of position is a constant, so we store it.
  Point _derivative;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Default constructor.  Uninitialized memory.
  ParametrizedLine() {}

  //! Construct from points.
  ParametrizedLine(const Point& source, const Point& target) :
    _segment(source, target),
    _derivative(_segment.getLength() * _segment.getTangent())
  {
    assert(isValid());
  }

  //! Make from points.
  void
  build(const Point& source, const Point& target)
  {
    _segment.make(source, target);
    _derivative = _segment.getLength() * _segment.getTangent();
    assert(isValid());
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  //@{

  //! Compute a point on the line from parameter values.
  Point
  computePosition(const ParameterPoint& parameterPoint) const
  {
    return _segment.getSource() +
           (parameterPoint[0] * _segment.getLength()) * _segment.getTangent();
  }

  //! Compute the derivative of position with respect to the parametrization.
  const Point&
  computeDerivative(const ParameterPoint& /*parameterPoint*/) const
  {
    return _derivative;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality operators.
  //@{

  //! Return true if the lines are equal.
  bool
  operator==(const ParametrizedLine& other)
  {
    return _segment == other._segment && _derivative == other._derivative;
  }


  //! Return true if the lines are not equal.
  bool
  operator!=(const ParametrizedLine& other)
  {
    return ! operator==(other);
  }

  //@}
  //--------------------------------------------------------------------------
  // Private member functions.

private:

  // Check that the length is non-zero.
  bool
  isValid()
  {
    return _segment.getLength() != 0;
  }
};


} // namespace geom
}

#endif

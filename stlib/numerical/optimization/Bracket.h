// -*- C++ -*-

/*!
  \file numerical/optimization/Bracket.h
  \brief Base class for bracketing methods.
*/

#if !defined(__numerical_optimization_Bracket_h__)
#define __numerical_optimization_Bracket_h__

#include <algorithm>
#include <array>
#include <limits>

#include <cmath>

namespace stlib
{
namespace numerical
{

//! Base class for bracketing methods.
template<class _Function>
class Bracket
{

  //
  // Member variables.
  //
protected:
  //! The objective function.
  _Function& _function;
private:
  std::array<double, 3> _x;
  std::array<double, 3> _f;


public:

  //! Construct from the objective function.
  Bracket(_Function& function) :
    _function(function),
    _x(),
    _f()
  {
    // Initialize with junk.
    std::fill(_x.begin(), _x.end(), std::numeric_limits<double>::max());
    std::fill(_f.begin(), _f.end(), std::numeric_limits<double>::max());
  }

  //! Bracket the function minima.
  /*! Given points \c a and \c b, search in the downhill
    direction and find a triple of ordered points that bracket a
    minimum of the function. The function value at the middle point is less
    than or equal to the other two function values. */
  void
  bracket(double a, double b);

  //! Return the triple of ordered points.
  const std::array<double, 3>&
  points() const
  {
    return _x;
  }

  //! Return the function values at the triple of ordered points.
  const std::array<double, 3>&
  values() const
  {
    return _f;
  }

  //! Return true if the function is properly bracketed.
  bool
  isValid() const
  {
    return _x[0] <= _x[1] && _x[1] <= _x[2] &&
           _f[1] <= _f[0] && _f[1] <= _f[2];
  }

protected:

  //! Shift the values.
  void
  shift(double* a, double* b, const double c)
  {
    *a = *b;
    *b = c;
  }

  //! Shift the values.
  void
  shift(double* a, double* b, double* c, const double d)
  {
    *a = *b;
    *b = *c;
    *c = d;
  }

  //! Move the last three to the first three.
  void
  move(double* a, double* b, double* c, const double d, const double e,
       const double f)
  {
    *a = d;
    *b = e;
    *c = f;
  }

private:

  //! Order the points in ascending order.
  void
  order()
  {
    if (_x[0] > _x[2]) {
      std::swap(_x[0], _x[2]);
      std::swap(_f[0], _f[2]);
    }
  }

};


} // namespace numerical
}

#define __numerical_optimization_Bracket_ipp__
#include "stlib/numerical/optimization/Bracket.ipp"
#undef __numerical_optimization_Bracket_ipp__

#endif

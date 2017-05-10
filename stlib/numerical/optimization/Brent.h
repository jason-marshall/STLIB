// -*- C++ -*-

/*!
  \file numerical/optimization/Brent.h
  \brief Base class for bracketing methods.
*/

#if !defined(__numerical_optimization_Brent_h__)
#define __numerical_optimization_Brent_h__

#include "stlib/numerical/optimization/Bracket.h"

#include <stdexcept>

namespace stlib
{
namespace numerical
{

//! The Brent method.
template<class _Function>
class Brent :
  Bracket<_Function>
{
  //
  // Types.
  //
private:

  typedef Bracket<_Function> Base;

  //
  // Member variables.
  //
private:

  const double _tolerance;
  using Base::_function;

public:

  //! Construct from the function and the tolerance.
  Brent(_Function& function, const double tolerance = 3.0e-8) :
    Base(function),
    _tolerance(tolerance)
  {
  }

  //! Minimize the function. Return the minimum value.
  double
  minimize(double a, double b, double* minPoint);

private:

  using Base::shift;
};


} // namespace numerical
}

#define __numerical_optimization_Brent_ipp__
#include "stlib/numerical/optimization/Brent.ipp"
#undef __numerical_optimization_Brent_ipp__

#endif

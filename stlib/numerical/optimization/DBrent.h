// -*- C++ -*-

/*!
  \file numerical/optimization/DBrent.h
  \brief Brent method that uses derivative information.
*/

#if !defined(__numerical_optimization_DBrent_h__)
#define __numerical_optimization_DBrent_h__

#include "stlib/numerical/optimization/Bracket.h"

#include <stdexcept>

namespace stlib
{
namespace numerical
{

//! Brent method that uses derivative information.
template<class _Function>
class DBrent :
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
  DBrent(_Function& function, const double tolerance = 3.0e-8) :
    Base(function),
    _tolerance(tolerance)
  {
  }

  //! Minimize the function. Return the minimum value.
  double
  minimize(double a, double b, double* minPoint);

private:

  using Base::move;
};


} // namespace numerical
}

#define __numerical_optimization_DBrent_ipp__
#include "stlib/numerical/optimization/DBrent.ipp"
#undef __numerical_optimization_DBrent_ipp__

#endif

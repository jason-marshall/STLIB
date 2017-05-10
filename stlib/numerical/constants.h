// -*- C++ -*-

/*!
  \file numerical/constants.h
  \brief Mathematical constants.
*/

#if !defined(__numerical_constants_h__)
#define __numerical_constants_h__

#include "stlib/numerical/constants/Exponentiation.h"
#include "stlib/numerical/constants/Logarithm.h"
#include "stlib/numerical/constants/pow.h"

namespace stlib
{
namespace numerical
{


/*!
  \page numerical_constants Mathematical Constants Package.

  The Constants structure defines some simple mathematical constants.
  - Constants<double>
  - Constants<float>
  .
  The Exponentiation class statically computes \f$B^E\f$ for non-negative,
  integer exponents.
*/

//! Mathematical constants.
template<typename T>
struct Constants;

//! Mathematical constants for the \c double type.
template<>
struct Constants<double> {

  //! \f$\pi = 3.14159\ldots\f$
  static
  double
  Pi()
  {
    return 3.1415926535897932384626433832795;
  }

  //! Base for the natural logarithm.  \f$e = 2.71828\ldots\f$
  static
  double
  E()
  {
    return 2.7182818284590452353602874713527;
  }

  //! Conversion from degrees to radians: \f$\pi / 180\f$.
  static
  double
  Degree()
  {
    return 0.017453292519943295769236907684886;
  }

  //! Conversion from degrees to radians.
  static
  double
  radians(const double degrees)
  {
    return degrees * Degree();
  }

  //! Conversion from radians to degrees.
  static
  double
  degrees(const double radians)
  {
    return radians * (1. / Degree());
  }
};


//! Mathematical constants for the \c float type.
template<>
struct Constants<float> {

  //! \f$\pi = 3.14159\ldots\f$
  static
  float
  Pi()
  {
    return 3.1415926535897932384626433832795f;
  }

  //! Base for the natural logarithm.  \f$e = 2.71828\ldots\f$
  static
  float
  E()
  {
    return 2.7182818284590452353602874713527f;
  }

  //! Conversion from degrees to radians: \f$\pi / 180\f$.
  static
  float
  Degree()
  {
    return 0.017453292519943295769236907684886f;
  }

  //! Conversion from degrees to radians.
  static
  float
  radians(const float degrees)
  {
    return degrees * Degree();
  }

  //! Conversion from radians to degrees.
  static
  float
  degrees(const float radians)
  {
    return radians * (1. / Degree());
  }
};


} // namespace numerical
}

#endif

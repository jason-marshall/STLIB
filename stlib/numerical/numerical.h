// -*- C++ -*-

#if !defined(__numerical_h__)
#define __numerical_h__

#include "stlib/numerical/constants.h"
#include "stlib/numerical/derivative.h"
#include "stlib/numerical/equality.h"
#include "stlib/numerical/grid_interp_extrap.h"
#include "stlib/numerical/interpolation.h"
#include "stlib/numerical/optimization.h"
#include "stlib/numerical/partition.h"
#include "stlib/numerical/polynomial.h"
#include "stlib/numerical/random.h"
#include "stlib/numerical/specialFunctions.h"

namespace stlib
{
//! All classes and functions in the Numerical Algorithms package are defined in the numerical namespace.
namespace numerical
{

/*!
\mainpage Numerical Algorithms Package
\anchor numerical

\section numerical_introduction Introduction

This is a %numerical algorithms package that I use in various
projects. It's not a general purpose library. I just add
functionality as I need it.

This is a templated C++ class library. All the functionality is
implemented in header files. Thus there is no library to compile or
link with. Just include the appropriate header files in your
application code when you compile.

This package is composed of a number of sub-packages. All classes
and functions are in the \c numerical namespace.
- The \ref numerical_constants "mathematical constants" micro-package has
  constants for \f$\pi\f$, Euler's constant e, and the like.
- There are functions for testing the
  \ref numericalEquality "approximate equality" of floating-point numbers.
- The \ref derivative "derivative" package has functions and functors for
  evaluating derivatives.
- The \ref grid_interp_extrap "grid interpolation/extrapolation" package
  is useful for interpolating field values in level-set applications.
- The \ref interpolation "interpolation package" has a variety of
  functions for performing polynomial interpolation.
- The \ref numerical_integer package has utilities for working with integers
  and bits.
- The \ref optimization "optimization" package
  implements a variety of optimization algorithms.
- The \ref partition "partition" micro-package has a function for
  fair partitioning of an integer.
- The \ref numerical_polynomial "polynomial" package has a function for
  evaluating polynomials.
- The \ref numerical_random "random number" package has functors for
  generating random numbers.
- The \ref numerical_specialFunctions "special functions" package has
  functors for \f$\Gamma\f$ and the like.
*/

} // namespace numerical
}

#endif

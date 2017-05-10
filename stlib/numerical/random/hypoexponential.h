// -*- C++ -*-

/*!
  \file numerical/random/hypoexponential.h
  \brief Includes the hypoexponential distribution classes.
*/

#if !defined(__numerical_random_hypoexponential_h__)
//! Include guard.
#define __numerical_random_hypoexponential_h__

#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionDistinctDynamicMinimumParameters.h"
#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionNormalApproximation.h"

namespace stlib
{
namespace numerical
{

/*!
  \page numerical_random_hypoexponential Hypoexponential Distribution Package

  The HypoexponentialDistributionDistinctDynamicMinimumParameters class
  calculates the hypoexponential distribution for a distinct set of rate
  parameters. It is used for certain trajectory tree methods in the
  stochastic package.

  The HypoexponentialDistributionNormalApproximation class implements a
  normal approximation of the hypoexponential distribution.
*/

#endif

} // namespace numerical
}

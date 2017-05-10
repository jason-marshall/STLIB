// -*- C++ -*-

/*!
  \file optimization/staticDimension.h
  \brief Includes the optimization classes for function of a specified dimension.
*/

/*!
  \page optimizationStaticDimension Optimization for Statically Specified Dimension.

  The numerical::QuasiNewton class implements the BFGS quasi-Newton method.

  The numerical::PenaltyQuasiNewton class implements the penalty method for
  equality constrained optimization.  It uses the
  numerical::FunctionWithQuadraticPenalty class with the quasi-Newton method.

  The numerical::Simplex class implements the downhill simplex method.

  The numerical::Penalty class implements the penalty method for equality
  constrained optimization.  It uses the
  numerical::FunctionWithQuadraticPenalty class with the downhill simplex
  method.

  The numerical::CoordinateDescent class implements the coordinate descent
  method of Hooke and Jeeves.

  Use this optimization package by including the file optimization/staticDimension.h.
*/

#if !defined(__numerical_optimization_staticDimension_h__)
#define __numerical_optimization_staticDimension_h__

#include "stlib/numerical/optimization/staticDimension/Simplex.h"
#include "stlib/numerical/optimization/staticDimension/CoordinateDescent.h"
#include "stlib/numerical/optimization/staticDimension/QuasiNewton.h"
#include "stlib/numerical/optimization/staticDimension/Penalty.h"
#include "stlib/numerical/optimization/staticDimension/PenaltyQuasiNewton.h"

#endif

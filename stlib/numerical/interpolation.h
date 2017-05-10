// -*- C++ -*-

/*!
  \file interpolation.h
  \brief Includes the interpolation functions.
*/

/*!
  \page interpolation Interpolation Package

  - There are several functors for performing
  \ref interpolation_PolynomialInterpolationUsingCoefficients "polynomial interpolation"
  on a 1-D regular grid. Because these store the polynomial coefficients
  they are fast, but use more memory than methods that work directly on the
  grid of function values.
  - The numerical::QuinticInterpolation2D class performs quintic
  interpolation on 2-D grids (period or plain). It stores the gradient
  and Hessian at the grid points in order to accelerate the
  interpolation.
  - There are several functors for performing linear and cubic
  \ref interpolation_InterpolatingFunction1DRegularGrid "interpolation on a 1-D regular grid".
  - There are also functors for performing
  \ref interpolation_InterpolatingFunctionRegularGrid "interpolation on an multi-dimensional regular grid".
  Constant value,
  multi-linear, and multi-cubic interpolation is supported for plain
  and periodic data.
  - \ref interpolation_simplex
  - \ref interpolation_hermite
  - The numerical::LinInterpGrid class is a functor for performing
    linear interpolation on a regular grid.
  - The numerical::PolynomialInterpolationNonNegative class may be useful
  for interpolating non-negative, nearly singular data.

  Use the interpolation package by including the file interpolation.h.
*/

#if !defined(__numerical_interpolation_h__)
#define __numerical_interpolation_h__

#include "stlib/numerical/interpolation/hermite.h"
#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGrid.h"
#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGridReference.h"
#include "stlib/numerical/interpolation/InterpolatingFunctionMultiArrayOf1DRegularGrids.h"
#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGrid.h"
#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGridReference.h"
#include "stlib/numerical/interpolation/InterpolatingFunctionMultiArrayOfRegularGrids.h"
#include "stlib/numerical/interpolation/LinInterpGrid.h"
#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficients.h"
#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsReference.h"
#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsVector.h"
#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsMultiArray.h"
#include "stlib/numerical/interpolation/simplex.h"

#endif

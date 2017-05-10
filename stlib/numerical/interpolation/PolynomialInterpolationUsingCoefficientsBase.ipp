// -*- C++ -*-

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficientsBase_ipp__)
#error This file is an implementation detail of PolynomialInterpolationUsingCoefficientsBase.
#endif

namespace stlib
{
namespace numerical
{

//--------------------------------------------------------------------------
// Constructors etc.

// Construct from the grid size and the Cartesian domain.
template<typename _T, std::size_t _Order>
inline
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
PolynomialInterpolationUsingCoefficientsBase(std::size_t gridSize,
    double lower, double upper) :
  _coefficients(0),
  // The number of cells is one less than the number of grid elements.
  _numberOfCells(gridSize - 1),
  _lowerCorner(lower),
  _inverseWidth(_numberOfCells / (upper - lower))
{
}

// Deep copy. The grids must be the same size.
template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
copy(const PolynomialInterpolationUsingCoefficientsBase& other)
{
  assert(this != &other);
  copyCoefficients(other);
  // No need to copy _numberOfCells.
  _lowerCorner = other._lowerCorner;
  _inverseWidth = other._inverseWidth;
}

// Copy the cell coefficients. The grids must be the same size.
template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
copyCoefficients(const PolynomialInterpolationUsingCoefficientsBase& other)
{
  assert(other._numberOfCells == _numberOfCells);
  std::copy(other._coefficients, other._coefficients + _numberOfCells,
            _coefficients);
}

//--------------------------------------------------------------------------
// Interpolation.

// Interpolate the field at the specified point.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
operator()(argument_type x) const
{
  return interpolate<_Order>(x);
}

// Interpolate the function and its derivative.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
operator()(argument_type x, Value* derivative) const
{
  return interpolate<_Order>(x, derivative);
}

// Interpolate the function and its first and second derivatives.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
operator()(argument_type x, Value* firstDerivative, Value* secondDerivative)
const
{
  return interpolate<_Order>(x, firstDerivative, secondDerivative);
}

// Interpolate the field at the specified point.
template<typename _T, std::size_t _Order>
template<std::size_t _InterpolationOrder>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(argument_type x) const
{
  static_assert(_InterpolationOrder <= _Order, "Not supported.");
  // Convert to index coordinates.
  x = _inverseWidth * (x - _lowerCorner);
#ifdef STLIB_DEBUG
  assert(x >= 0);
#endif
  // Convert to std::size_t to obtain the cell index.
  const std::size_t i = std::size_t(x);
#ifdef STLIB_DEBUG
  assert(i < _numberOfCells);
#endif
  x -= i;
  return interpolate(x, i, std::integral_constant<std::size_t, _Order>());
}


// Interpolate the function and its derivative.
template<typename _T, std::size_t _Order>
template<std::size_t _InterpolationOrder>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(argument_type x, Value* derivative) const
{
  static_assert(_InterpolationOrder <= _Order, "Not supported.");
  static_assert(_Order >= 3, "Not supported.");
  // Convert to index coordinates.
  x = _inverseWidth * (x - _lowerCorner);
#ifdef STLIB_DEBUG
  assert(x >= 0);
#endif
  // Convert to std::size_t to obtain the cell index.
  const std::size_t i = std::size_t(x);
#ifdef STLIB_DEBUG
  assert(i < _numberOfCells);
#endif
  x -= i;
  return interpolate(x, i, derivative,
                     std::integral_constant<std::size_t, _Order>());
}


// Interpolate the function and its first and second derivatives.
template<typename _T, std::size_t _Order>
template<std::size_t _InterpolationOrder>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(argument_type x, Value* firstDerivative, Value* secondDerivative)
const
{
  static_assert(_InterpolationOrder == _Order, "Not supported.");
  static_assert(_Order == 5, "Not supported.");
  // Convert to index coordinates.
  x = _inverseWidth * (x - _lowerCorner);
#ifdef STLIB_DEBUG
  assert(x >= 0);
#endif
  // Convert to std::size_t to obtain the cell index.
  const std::size_t i = std::size_t(x);
#ifdef STLIB_DEBUG
  assert(i < _numberOfCells);
#endif
  x -= i;
  return interpolate(x, i, firstDerivative, secondDerivative);
}

// Use the grid values to set the polynomial coefficient values.
template<typename _T, std::size_t _Order>
template<typename _ForwardIterator>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setGridValues(_ForwardIterator f)
{
  // The function values plus two guards.
  std::vector<Value> fg(_numberOfCells + 5);
  // Copy the values.
  _ForwardIterator end = f;
  std::advance(end, _numberOfCells + 1);
  std::copy(f, end, &fg[2]);
  // We need guard values on both ends of the regular grid.
  if (_numberOfCells >= 2) {
    fg[1] = 3 * fg[2] - 3 * fg[3] + fg[4];
    fg[0] = 3 * fg[1] - 3 * fg[2] + fg[3];
    fg[_numberOfCells + 3] = fg[_numberOfCells] - 3 * fg[_numberOfCells + 1]
                             + 3 * fg[_numberOfCells + 2];
    fg[_numberOfCells + 4] = fg[_numberOfCells + 1] - 3 * fg[_numberOfCells + 2]
                             + 3 * fg[_numberOfCells + 3];
  }
  else {
    fg[1] = 2 * fg[2] - fg[3];
    fg[0] = 3 * fg[2] - 2 * fg[3];
    fg[_numberOfCells + 3] = 2 * fg[_numberOfCells + 2] - fg[_numberOfCells + 1];
    fg[_numberOfCells + 4] = 3 * fg[_numberOfCells + 2]
                             - 2 * fg[_numberOfCells + 1];
  }
  // Set the polynomial coefficients.
  setCoefficients(&fg[2], std::integral_constant<std::size_t, _Order>());
}

// Use the grid values to set the polynomial coefficient values.
template<typename _T, std::size_t _Order>
template<typename _ForwardIterator>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setGridValues(_ForwardIterator f, _ForwardIterator df)
{
  // The function values plus guards.
  std::vector<Value> fg(_numberOfCells + 3);
  // Copy the values.
  _ForwardIterator end = f;
  std::advance(end, _numberOfCells + 1);
  std::copy(f, end, &fg[1]);

  // The derivative values plus guards.
  std::vector<Value> dfg(_numberOfCells + 3);
  // Copy the values.
  end = df;
  std::advance(end, _numberOfCells + 1);
  std::copy(df, end, &dfg[1]);
  // Scale to go from physical coordinates to logical coordinates.
  const double scale = 1. / _inverseWidth;
  for (std::size_t i = 0; i != dfg.size(); ++i) {
    dfg[i] *= scale;
  }

  // We need guard values on both ends of the regular grid.
  if (_numberOfCells >= 2) {
    fg[0] = -18 * fg[1] + 9 * fg[2] + 10 * fg[3]
            - 9 * dfg[1] - 18 * dfg[2] - 3 * dfg[3];
    fg[_numberOfCells + 2] =
      -18 * fg[_numberOfCells + 1] + 9 * fg[_numberOfCells]
      + 10 * fg[_numberOfCells - 1]
      + 9 * dfg[_numberOfCells + 1] + 18 * dfg[_numberOfCells]
      + 3 * dfg[_numberOfCells - 1];
    dfg[0] = 57 * fg[1] - 24 * fg[2] - 33 * fg[3]
             + 24 * dfg[1] + 57 * dfg[2] + 10 * dfg[3];
    dfg[_numberOfCells + 2] =
      -57 * fg[_numberOfCells + 1] + 24 * fg[_numberOfCells]
      + 33 * fg[_numberOfCells - 1]
      + 24 * dfg[_numberOfCells + 1] + 57 * dfg[_numberOfCells]
      + 10 * dfg[_numberOfCells - 1];
  }
  else {
    fg[0] = -4 * fg[1] + 5 * fg[2] - 4 * dfg[1] - 2 * dfg[2];
    fg[_numberOfCells + 2] =
      -4 * fg[_numberOfCells + 1] + 5 * fg[_numberOfCells]
      + 4 * dfg[_numberOfCells + 1] + 2 * dfg[_numberOfCells];
    dfg[0] = 12 * fg[1] - 12 * fg[2] + 8 * dfg[1] + 5 * dfg[2];
    dfg[_numberOfCells + 2] =
      -12 * fg[_numberOfCells + 1] + 12 * fg[_numberOfCells]
      + 8 * dfg[_numberOfCells + 1] + 5 * dfg[_numberOfCells];
  }

  // Set the polynomial coefficients.
  setCoefficients(&fg[1], &dfg[1], std::integral_constant<std::size_t, _Order>());
}

// Use the grid values to set the polynomial coefficient values.
template<typename _T, std::size_t _Order>
template<typename _ForwardIterator>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setGridValues(_ForwardIterator f, _ForwardIterator df,
              _ForwardIterator ddf)
{
#if 0
  // The function values plus guards.
  std::vector<Value> fg(_numberOfCells + 3);
  // Copy the values.
  _ForwardIterator end = f;
  std::advance(end, _numberOfCells + 1);
  std::copy(f, end, &fg[1]);

  // The first derivative values plus guards.
  std::vector<Value> dfg(_numberOfCells + 3);
  // Copy the values.
  end = df;
  std::advance(end, _numberOfCells + 1);
  std::copy(df, end, &dfg[1]);
  // Scale to go from physical coordinates to logical coordinates.
  {
    const double scale = 1. / _inverseWidth;
    for (std::size_t i = 0; i != dfg.size(); ++i) {
      dfg[i] *= scale;
    }
  }

  // The second derivative values plus guards.
  std::vector<Value> ddfg(_numberOfCells + 3);
  // Copy the values.
  end = ddf;
  std::advance(end, _numberOfCells + 1);
  std::copy(ddf, end, &ddfg[1]);
  // Scale to go from physical coordinates to logical coordinates.
  {
    const double scale = 1. / (_inverseWidth * _inverseWidth);
    for (std::size_t i = 0; i != ddfg.size(); ++i) {
      ddfg[i] *= scale;
    }
  }

  // We need guard values on both ends of the regular grid.
  // Quadratic extrapolation using the closest point.
  fg[0] = fg[1] - dfg[1] + 0.5 * ddfg[1];
  fg[_numberOfCells + 2] = fg[_numberOfCells + 1] + dfg[_numberOfCells + 1]
                           + 0.5 * ddfg[_numberOfCells + 1];
  // Linear extrapolation using the closest point.
  dfg[0] = dfg[1] - ddfg[1];
  dfg[_numberOfCells + 2] = dfg[_numberOfCells + 1] + ddfg[_numberOfCells + 1];
  // Linear extrapolation using the two closest points.
  ddfg[0] = 2 * ddfg[1] - ddfg[2];
  ddfg[_numberOfCells + 2] = 2 * ddfg[_numberOfCells + 1]
                             - ddfg[_numberOfCells];

  // Set the polynomial coefficients.
  setCoefficients(&fg[1], &dfg[1], &ddfg[1],
                  std::integral_constant<std::size_t, _Order>());
#endif
  //
  // Note that there is no need for guard points.
  //
  // The function values.
  std::vector<Value> fg(_numberOfCells + 1);
  // Copy the values.
  _ForwardIterator end = f;
  std::advance(end, _numberOfCells + 1);
  std::copy(f, end, &fg[0]);

  // The first derivative values.
  std::vector<Value> dfg(_numberOfCells + 1);
  // Copy the values.
  end = df;
  std::advance(end, _numberOfCells + 1);
  std::copy(df, end, &dfg[0]);
  // Scale to go from physical coordinates to logical coordinates.
  {
    const double scale = 1. / _inverseWidth;
    for (std::size_t i = 0; i != dfg.size(); ++i) {
      dfg[i] *= scale;
    }
  }

  // The second derivative values.
  std::vector<Value> ddfg(_numberOfCells + 1);
  // Copy the values.
  end = ddf;
  std::advance(end, _numberOfCells + 1);
  std::copy(ddf, end, &ddfg[0]);
  // Scale to go from physical coordinates to logical coordinates.
  {
    const double scale = 1. / (_inverseWidth * _inverseWidth);
    for (std::size_t i = 0; i != ddfg.size(); ++i) {
      ddfg[i] *= scale;
    }
  }

  // Set the polynomial coefficients.
  setCoefficients(&fg[0], &dfg[0], &ddfg[0],
                  std::integral_constant<std::size_t, _Order>());
}

// Linear interpolation.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            std::integral_constant<std::size_t, 1> /*Order*/) const
{
  const Coefficients& c = _coefficients[i];
  return c[0] + c[1] * x;
}

// Cubic interpolation.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            std::integral_constant<std::size_t, 3> /*Order*/) const
{
  const Coefficients& c = _coefficients[i];
  return c[0] + x * (c[1] + x * (c[2] + x * c[3]));
}

// Cubic interpolation with derivative.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            Value* derivative,
            std::integral_constant<std::size_t, 3> /*Order*/) const
{
  const Coefficients& c = _coefficients[i];
  *derivative = c[1] + x * (2 * c[2] + x * 3 * c[3]);
  // Scale to go from logical coordinates to physical coordinates.
  *derivative *= _inverseWidth;
  return c[0] + x * (c[1] + x * (c[2] + x * c[3]));
}

// Quintic interpolation.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            std::integral_constant<std::size_t, 5> /*Order*/) const
{
  const Coefficients& c = _coefficients[i];
  return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4]
                                 + x * c[5]))));
}

// Quintic interpolation with first derivative.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            Value* derivative,
            std::integral_constant<std::size_t, 5> /*Order*/) const
{
  const Coefficients& c = _coefficients[i];
  *derivative = c[1]
                + x * (2 * c[2] + x * (3 * c[3] + x * (4 * c[4] + x * 5 * c[5])));
  // Scale to go from logical coordinates to physical coordinates.
  *derivative *= _inverseWidth;
  return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4]
                                 + x * c[5]))));
}

// Quintic interpolation with first and second derivatives.
template<typename _T, std::size_t _Order>
inline
typename PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::result_type
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
interpolate(const argument_type x, const std::size_t i,
            Value* firstDerivative, Value* secondDerivative) const
{
  const Coefficients& c = _coefficients[i];
  *firstDerivative = c[1]
                     + x * (2 * c[2] + x * (3 * c[3] + x * (4 * c[4] + x * 5 * c[5])));
  *secondDerivative = 2 * c[2]
                      + x * (6 * c[3] + x * (12 * c[4] + x * 20 * c[5]));
  // Scale to go from logical coordinates to physical coordinates.
  *firstDerivative *= _inverseWidth;
  *secondDerivative *= _inverseWidth * _inverseWidth;
  return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4]
                                 + x * c[5]))));
}

// In the following setCoefficients() functions use std::ptrdiff_t because
// we may access f[-1].

//
// Function.
//

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f,
                std::integral_constant<std::size_t, 1> /*Order*/)
{
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
    c[0] = f[i];
    c[1] = - f[i] + f[i + 1];
  }
  checkErrors();
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f,
                std::integral_constant<std::size_t, 3> /*Order*/)
{
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
    c[0] = f[i];
    c[1] = 0.5 * (- f[i - 1] + f[i + 1]);
    c[2] = 0.5 * (2 * f[i - 1] - 5 * f[i] + 4 * f[i + 1] - f[i + 2]);
    c[3] = 0.5 * (-f[i - 1] + 3 * f[i] - 3 * f[i + 1] + f[i + 2]);
  }
  checkErrors();
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f,
                std::integral_constant<std::size_t, 5> /*Order*/)
{
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
#if 0
    c[0] = f[i];
    c[1] = 0.5 * (- f[i - 1] + f[i + 1]);
    c[2] = 0.5 * (f[i - 1] - 2 * f[i] + f[i + 1]);
    c[3] = 1.5 * (f[i - 1] - 3 * f[i] + 3 * f[i + 1] - f[i + 2]);
    c[4] = 2.5 * (- f[i - 1] + 3 * f[i] - 3 * f[i + 1] + f[i + 2]);
    c[5] = f[i - 1] - 3 * f[i] + 3 * f[i + 1] - f[i + 2];
#else
    c[0] = f[i];
    c[1] = (1. / 12.) * (f[-2 + i] - 8 * f[-1 + i] + 8 * f[1 + i]
                         - f[2 + i]);
    c[2] = (1. / 24.) * (-f[-2 + i] + 16 * f[-1 + i] - 30 * f[i]
                         + 16 * f[1 + i] - f[2 + i]);
    c[3] = (1. / 24.) * (-9 * f[-2 + i] + 39 * f[-1 + i] - 70 * f[i]
                         + 66 * f[1 + i] - 33 * f[2 + i] + 7 * f[3 + i]);
    c[4] = (1. / 24.) * (13 * f[-2 + i] - 64 * f[-1 + i] + 126 * f[i]
                         - 124 * f[1 + i] + 61 * f[2 + i] - 12 * f[3 + i]);
    c[5] = (-5. / 24.) * (f[-2 + i] - 5 * f[-1 + i] + 10 * f[i]
                          - 10 * f[1 + i] + 5 * f[2 + i] - f[3 + i]);
#endif
  }
  checkErrors();
}

//
// Function and first derivative.
//

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* /*df*/,
                std::integral_constant<std::size_t, 1> /*Order*/)
{
  // Ignore the derivative information.
  setCoefficients(f, std::integral_constant<std::size_t, 1>());
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* df,
                std::integral_constant<std::size_t, 3> /*Order*/)
{
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
    c[0] = f[i];
    c[1] = df[i];
    c[2] = - 3 * f[i] + 3 * f[1 + i] - 2 * df[i] - df[1 + i];
    c[3] = 2 * f[i] - 2 * f[i + 1] + df[i] + df[i + 1];
  }
  checkErrors();
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* df,
                std::integral_constant<std::size_t, 5> /*Order*/)
{
  // Use f and df for second derivative.
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
    c[0] = f[i];
    c[1] = df[i];
    c[2] = 0.25 * (df[i - 1] - df[i + 1] + 4 * f[i - 1] - 8 * f[i] + 4 * f[i + 1]);
    c[3] = 0.25 * (-3 * df[i - 1] - 23 * df[i] - 13 * df[i + 1] - df[i + 2]
                   - 12 * f[i - 1] - 12 * f[i] + 20 * f[i + 1] + 4 * f[i + 2]);
    c[4] = 0.25 * (3 * df[i - 1] + 30 * df[i] + 25 * df[i + 1] + 2 * df[i + 2]
                   + 12 * f[i - 1] + 28 * f[i] - 32 * f[i + 1] - 8 * f[i + 2]);
    c[5] = 0.25 * (-df[i - 1] - 11 * df[i] - 11 * df[i + 1] - df[i + 2]
                   - 4 * f[i - 1] - 12 * f[i] + 12 * f[i + 1] + 4 * f[i + 2]);
  }
  checkErrors();
}

//
// Function, first derivative, and second derivative.
//

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* /*df*/, const Value* /*ddf*/,
                std::integral_constant<std::size_t, 1> /*Order*/)
{
  // Ignore the derivative information.
  setCoefficients(f, std::integral_constant<std::size_t, 1>());
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* df, const Value* /*ddf*/,
                std::integral_constant<std::size_t, 3> /*Order*/)
{
  // Ignore the second derivative information.
  setCoefficients(f, df, std::integral_constant<std::size_t, 3>());
}

template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
setCoefficients(const Value* f, const Value* df, const Value* ddf,
                std::integral_constant<std::size_t, 5> /*Order*/)
{
  for (std::ptrdiff_t i = 0; i != std::ptrdiff_t(_numberOfCells); ++i) {
    Coefficients& c = _coefficients[i];
    c[0] = f[i];
    c[1] = df[i];
    c[2] = 0.5 * ddf[i];
    c[3] =  0.5 * (20 * (-f[i] + f[i + 1]) - 12 * df[i] - 8 * df[i + 1]
                   - 3 * ddf[i] + ddf[i + 1]);
    c[4] =  0.5 * (30 * (f[i] - f[i + 1]) + 16 * df[i] + 14 * df[i + 1]
                   + 3 * ddf[i] - 2 * ddf[i + 1]);
    c[5] =  0.5 * (12 * (-f[i] + f[i + 1]) - 6 * (df[i] + df[i + 1])
                   - ddf[i] + ddf[i + 1]);
  }
  checkErrors();
}


template<typename _T, std::size_t _Order>
inline
void
PolynomialInterpolationUsingCoefficientsBase<_T, _Order>::
checkErrors() const
{
  // Don't test when using GCC 4.4 or previous.
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 4)
  // The threshold is the square root of the machine epsilon.
  const result_type Threshold =
    std::sqrt(std::numeric_limits<result_type>::epsilon());
  // Check all except the last cell. (We don't have explicit information
  // for the function value at the end of the last cell.)
  for (std::size_t i = 0; i != _numberOfCells - 1; ++i) {
    const result_type f =
      interpolate(argument_type(1), i,
                  std::integral_constant<std::size_t, _Order>());
    result_type denominator = std::max(std::abs(_coefficients[i][0]),
                                       std::abs(_coefficients[i + 1][0]));
    if (denominator < std::numeric_limits<result_type>::epsilon()) {
      denominator = 1;
    }
    const result_type relativeError = std::abs(f - _coefficients[i + 1][0]) /
                                      denominator;
    if (relativeError > Threshold) {
      std::ostringstream stream;
      stream << "Error in PolynomialInterpolationUsingCoefficientsBase:\n"
             << "In cell " << i << ", the relative error " << relativeError
             << " exceeds the allowed threshold " << Threshold << ".\n"
             << "The function values are " << _coefficients[i][0]
             << " and " << _coefficients[i + 1][0] << ".";
      // This causes an error with the MacPorts version of GCC 4.4.
      // Specifically, throwing the runtime_error causes an abort; it is
      // not caught. This is a problem with GCC.
      throw std::runtime_error(stream.str().c_str());
    }
  }
#endif
}


} // namespace numerical
}

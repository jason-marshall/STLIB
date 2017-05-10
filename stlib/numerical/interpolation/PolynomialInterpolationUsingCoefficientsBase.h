// -*- C++ -*-

/*!
  \file numerical/interpolation/PolynomialInterpolationUsingCoefficientsBase.h
  \brief Base class for polynomial interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_PolynomialInterpolationUsingCoefficientsBase_h__)
#define __numerical_interpolation_PolynomialInterpolationUsingCoefficientsBase_h__

#include "stlib/ext/array.h"

#include <functional>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cassert>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

/*!
  \page interpolation_PolynomialInterpolationUsingCoefficients %Polynomial interpolation using coefficients.

  \par Introduction.
  <a href="http://en.wikipedia.org/wiki/Interpolation">Interpolation</a> is a
  method of evaluating a function given only its values (and perhaps derivative
  values) at a discrete set of points.
  <a href="http://en.wikipedia.org/wiki/Polynomial_interpolation">Piecewise
  polynomial interpolation</a> breaks the interpolation domain into a number
  of disjoint \e cells. Consider a 1-D function <em>y(x)</em> that is sampled
  at a set of points {<em>x<sub>i</sub></em>} and has the values
  <em>y<sub>i</sub> = y(x<sub>i</sub>)</em>.
  In <a href="http://en.wikipedia.org/wiki/Linear_interpolation">linear
  interpolation</a> the cells are the intervals
  [<em>x<sub>i</sub>..x<sub>i</sub></em>). The <em>i</em><sup>th</sup>
  <em>interpolant</em> is a linear function that passes through the
  two points (<em>x<sub>i</sub>, p<sub>i</sub></em>) and
  (<em>x<sub>i+1</sub>, p<sub>i+1</sub></em>).

  \par Higher order methods.
  A cubic polynomial has four coefficients:
  <em>a<sub>0</sub> + a<sub>1</sub> x + a<sub>2</sub> x<sup>2</sup>
  + a<sub>3</sub> x<sup>3</sup></em>. Thus four grids values are used
  to determine the polynomial. If only function values are available, then
  in addition to the grid points at the lower and upper boundaries of the cell,
  the directly adjacent values are used as well. Thus cubic interpolation
  uses a four point \e stencil. A quintic (fifth order) polynomial has six
  coefficients and thus has a six point stencil. (Quadratic, quartic, and
  other even-ordered interpolation is rarely used because the stencil is not
  symmetric with respect to the cell. This leads to less desirable properties
  than odd-ordered interpolants)

  \par Higher degree sampling.
  If only function values are available, we call this zeroth degree sampling.
  If first derivatives or first and second derivatives are also available then
  we have first and second degree sampling, respectively. As we will see later
  on supplying derivative information improves the accuracy of the interpolants.
  For example if one is performing cubic interpolation and first derivatives
  are available (order 3, degree 1) then each interpolant will use the
  function values and derivatives that are given at the lower and upper
  boundaries of the cell. This yields better results than using four
  function values because the stencil is more compact. That is, the information
  about the function is closer to the domain on which it will be used.

  \par Storing grids versus storing coefficients.
  The interpolation data structure may either store the grids that sample
  the function or it may store the polynomial coefficients. The latter
  method (which we use in the classes present below) has higher storage
  requirements, but has lower computational costs. Specifically, For
  order \e R interpolation with degree \e D data storing the grids
  requires <em>(D + 1)(N + 1)</em> floating-point number, where \e N is
  the number of cells. By comparison storing the polynomial coefficients
  requires <em>(R + 1) N</em> numbers. For the common case that
  <em>D = 2 R + 1</em> the polynomial coefficients require about twice the
  storage of the grids.


  \par Overview of classes.
  The functors numerical::PolynomialInterpolationUsingCoefficients and
  numerical::PolynomialInterpolationUsingCoefficientsReference
  both support 1<sup>st</sup> order (linear), 3<sup>rd</sup> order
  (cubic), and 5<sup>th</sup> (quintic) interpolation on regular
  (uniformly spaced) grids.
  The difference between these two classes is that
  the former allocates memory for the polynomial coefficients whereas the latter
  references externally allocated memory.
  The domain associated with the grid exactly bounds the grid points.

  \par Multiple grids.
  If one is performing interpolation on multiple grids that have the same
  size and Cartesian domain then it is probably convenient to use either
  numerical::PolynomialInterpolationUsingCoefficientsVector or
  numerical::PolynomialInterpolationUsingCoefficientsMultiArray.
  These classes manage the memory for a vector or a multi-dimensional
  %array of coefficients. (For the latter the %array dimension is a
  template parameter.)
  One selects the desired grid before interpolating.
  Using these classes is more efficient that allocating an %array
  of either numerical::PolynomialInterpolationUsingCoefficients or
  numerical::PolynomialInterpolationUsingCoefficientsReference because they do
  not store redundant information about the grids and because all of the data
  for the polynomial coefficients is stored in one contiguous %array.

  \par Using PolynomialInterpolationUsingCoefficients.
  In the code example below we construct a linear interpolating function.
  To construct a numerical::PolynomialInterpolationUsingCoefficients one
  specifies the function values sampled on a grid and the lower and upper
  bounds of the domain spanned by the grid. To perform interpolation use
  function call operator. That is if \c f is the functor then \c f(x)
  returns the interpolated value at the point \c x. One may also
  use the \c interpolate() member function with an explicitly specified
  interpolation order. Note that this order must be less than or equal
  to the order used when constructing the functor. For example
  \c f.interpolate<1>(x) returns the same result as \c f(x). This
  functionality is only useful when one wants to perform a lower order
  interpolation than what was specified when constructing the functor.

  \note If \e a and \e b are the positions of the first and last grid
  point, then one may only perform interpolation on points in the
  semi-open interval <em>[a..b)</em>. This is due to how the grid is organized
  into cells to perform the interpolation. Let there be \e N grid points with
  positions <em>g<sub>i</sub></em>. Then there are \e N - 1 cells, which
  are composed of adjacent grid points. The cell with index \e i is
  used to perform interpolation in the interval
  <em>[g<sub>i</sub>..g<sub>i+1</sub>)</em>. If debugging is enabled
  with the STLIB_DEBUG macro then trying to perform interpolation on
  a position that is outside the grid domain will result in an assertion
  failure. If debugging is not enabled then this will result in
  <em>undefined behavior</em>. Thus it is critical that the user ensure that
  the interpolation arguments are in the correct range.

  \code
  typedef numerical::PolynomialInterpolationUsingCoefficients<double, 1> F;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = std::exp(Dx * i);
  }
  // Construct the interpolating function.
  F f(grid.begin(), grid.size(), Lower, Upper);
  // Check that the function has the correct values at the grid points.
  // Note that we cannot evaluate the interpolating function at values
  // greater than or equal to the last grid point.
  const std::size_t numberOfCells = grid.size() - 1;
  for (std::size_t i = 0; i != numberOfCells; ++i) {
     assert(numerical::areEqual(f(Dx * i), grid[i]));
  }
  // Change the interpolating function to sample the function f(x) = x^2.
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = (Dx * i) * (Dx * i);
  }
  f.setGridValues(grid.begin());
  \endcode

  \par
  To use cubic interpolation one specifies an interpolation order of
  three as a template parameter when constructing the functor.
  The constructor arguments are the same as for linear interpolation.
  One may either evaluate the function or the function and its derivative.

  \code
  typedef numerical::PolynomialInterpolationUsingCoefficients<double, 3> F;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = std::exp(Dx * i);
  }
  // Construct the interpolating function.
  F f(grid.begin(), grid.size(), Lower, Upper);
  // Check that the function has reasonable values at the cell centers.
  const std::size_t numberOfCells = grid.size() - 1;
  double derivative;
  for (std::size_t i = 0; i != numberOfCells; ++i) {
     const double x = Dx * (i + 0.5);
     assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
     // Check the derivative as well.
     assert(std::abs(f(x, &derivative) - std::exp(x)) < Dx * Dx);
     assert(std::abs(derivative - std::exp(x)) < Dx);
  }
  \endcode

  \par
  For quintic (fifth order) interpolation one may evaluate the function,
  the function and its first derivative, or the function and its first
  and second derivatives.

  \code
  typedef numerical::PolynomialInterpolationUsingCoefficients<double, 5> F;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = std::exp(Dx * i);
  }
  // Construct the interpolating function.
  F f(grid.begin(), grid.size(), Lower, Upper);
  // Check that the function has reasonable values at the cell centers.
  const std::size_t numberOfCells = grid.size() - 1;
  double first, second;
  for (std::size_t i = 0; i != numberOfCells; ++i) {
     const double x = Dx * (i + 0.5);
     // Check the function value.
     assert(std::abs(f(x) - std::exp(x)) < Dx * Dx);
     // Check the first derivative as well.
     assert(std::abs(f(x, &first) - std::exp(x)) < Dx * Dx);
     assert(std::abs(first - std::exp(x)) < Dx);
     // Check the first and second derivatives.
     assert(std::abs(f(x, &first, &second) - std::exp(x)) < std::exp(1) * Dx * Dx);
     assert(std::abs(first - std::exp(x)) < std::exp(1) * Dx);
     assert(std::abs(second - std::exp(x)) < std::exp(1) * Dx);
  }
  \endcode

  \par Referencing grid memory.
  To construct a
  numerical::PolynomialInterpolationUsingCoefficientsReference one specifies
  a pointer to the coefficients data, the number of cells, and the Cartesian
  domain of the grid in the constructor. Below we show an example using
  linear interpolation.

  \code
  typedef numerical::PolynomialInterpolationUsingCoefficientsReference<double, 1> F;
  typedef F::Coefficients Coefficients;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = std::exp(Dx * i);
  }
  // Construct the interpolating function.
  std::vector<Coefficients> coefficients(grid.size() - 1);
  F f(&coefficients[0], coefficients.size(), Lower, Upper);
  f.setGridValues(grid.begin());
  // Check that the function has the correct values at the grid points.
  // Note that we cannot evaluate the interpolating function at values
  // greater than or equal to the last grid point.
  for (std::size_t i = 0; i != coefficients.size(); ++i) {
     assert(numerical::areEqual(f(Dx * i), grid[i]));
  }
  // Change the interpolating function to sample the function f(x) = x^2.
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = (Dx * i) * (Dx * i);
  }
  f.setGridValues(grid.begin());
  \endcode

  \par Using a vector of grids.
  To construct a
  numerical::PolynomialInterpolationUsingCoefficientsVector one specifies
  the grid size, the Cartesian domain, and the number of grids in the
  constructor. One selects a grid with setGrid(). One may then set the
  grid values with setGridValues() or perform interpolation with the
  standard interface.

  \code
  // Linear interpolation with a vector of grids.
  typedef numerical::PolynomialInterpolationUsingCoefficientsVector<double, 1> F;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  const std::size_t numberOfGrids = 20;
  // Construct the interpolating function.
  F f(grid.size(), Lower, Upper, numberOfGrids);
  // Set values for each of the grids.
  // Loop over the grids in the vector.
  for (std::size_t n = 0; n != f.getNumberOfGrids(); ++n) {
     // Select a grid.
     f.setGrid(n);
     // A different offset for each grid.
     const double offset = n;
     // Set the grid values.
     for (std::size_t i = 0; i != grid.size(); ++i) {
        grid[i] = offset + std::exp(Dx * i);
     }
     f.setGridValues(grid.begin());
  }
  // Check that the function has the correct values at the grid points.
  const std::size_t numberOfCells = grid.size() - 1;
  for (std::size_t n = 0; n != f.getNumberOfGrids(); ++n) {
     f.setGrid(n);
     const double offset = n;
     for (std::size_t i = 0; i != numberOfCells; ++i) {
        assert(numerical::areEqual(f(Dx * i), offset + std::exp(Dx * i)));
     }
  }
  \endcode

  \par Using a multi-dimensional %array of grids.
  To construct a
  numerical::PolynomialInterpolationUsingCoefficientsMultiArray one specifies
  the grid size, the Cartesian domain, and the %array extents in the
  constructor. One selects a grid with setGrid(). One may then set the
  grid values with setGridValues() or perform interpolation with the
  standard interface.

  \code
  // Linear interpolation. 2-D array of grids.
  const std::size_t Dimension = 2;
  typedef numerical::PolynomialInterpolationUsingCoefficientsMultiArray<double, 1, Dimension> F;
  typedef F::SizeList SizeList;
  typedef F::IndexList IndexList;
  typedef container::MultiIndexRange<Dimension> ArrayRange;
  typedef container::MultiIndexRangeIterator<Dimension> ArrayIterator;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::vector<double> grid(11);
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (grid.size() - 1);
  const SizeList arrayExtents = {{10, 20}};
  // Construct the interpolating function.
  F f(grid.size(), Lower, Upper, arrayExtents);
  // Set values for each of the grids.
  const ArrayRange arrayRange(arrayExtents);
  const ArrayIterator arrayBegin = ArrayIterator::begin(arrayRange);
  const ArrayIterator arrayEnd = ArrayIterator::end(arrayRange);
  // Loop over the grids in the array.
  for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
     // Select a grid.
     f.setGrid(*a);
     // A different offset for each grid.
     const double offset = sum(*a);
     // Set the grid values.
  for (std::size_t i = 0; i != grid.size(); ++i) {
     grid[i] = offset + std::exp(Dx * i);
  }
  f.setGridValues(grid.begin());
  }
  // Check that the function has the correct values at the grid points.
  const std::size_t numberOfCells = grid.size() - 1;
  for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
     f.setGrid(*a);
     const double offset = sum(*a);
  for (std::size_t i = 0; i != numberOfCells; ++i) {
     assert(numerical::areEqual(f(Dx * i), offset + std::exp(Dx * i)));
  }
  \endcode

  \par
  The interface for cubic and quintic interpolation is the same (except for
  the addition of the function call operators that compute derivatives.

  \par Feature comparison of the functors.
  As mentioned above the difference between
  numerical::PolynomialInterpolationUsingCoefficients and
  numerical::PolynomialInterpolationUsingCoefficientsReference
  is that the former allocates memory for the coefficients whereas the latter
  references externally allocated memory. Both have copy constructors,
  and assignment operators. Note however that to use assignment with
  the latter class the grids must be the same size. The former class is easier
  to construct, because one does not need to allocate memory for the
  polynomial coefficients.

  \par
  If one has many grids with the same size and Cartesian domain and
  if the grids can be logically arranged in a multi-dimensional %array then
  it is best to use either
  numerical::PolynomialInterpolationUsingCoefficientsVector or
  numerical::PolynomialInterpolationUsingCoefficientsMultiArray. These classes
  efficiently manage memory for the grids and allow the user to
  manipulate them.

  \par Acceptable errors.
  When the interpolating function is constructed, simple tests for checking
  the accuracy of the interpolant are performed. If the function values
  on the grid differ by many orders of magnitude, then the roundoff-error
  may produce inaccurate interpolant. In extreme cases, the error term will
  dominate. If the relative error exceeds the square root of the machine
  epsilon, then an exception will be thrown. One may either catch the
  exception, or allow it to terminate the program. Consult the unit tests
  for a example of catching the exception and printing a warning message.

  \par Accuracy.
  We examine the accuracy of the various methods by interpolating the
  function sin(\e x) on the interval \f$[0..\pi)\f$.
  We use 5 cells, which means
  that the function is sampled at 6 points. Below we show the function and
  the various interpolants.

  \image html PolynomialInterpolationSineFunction.jpg "Interpolating functions for sin(x)."
  \image latex PolynomialInterpolationSineFunction.pdf "Interpolating functions for sin(x)."

  \par
  Only linear interpolation is clearly different than the sampled function.
  To examine the accuracy we plot the errors in the interpolants below.
  Linear interpolation results in relatively large errors. Both cubic
  interpolation using first derivative information and quintic interpolation
  using second derivative information perform well.

  \image html PolynomialInterpolationSineError.jpg "Error in the interpolants"
  \image latex PolynomialInterpolationSineError.pdf "Error in the interpolants"

  \par
  Next we consider methods that use only the function values, and do not
  sample its derivatives. Again we see that linear interpolation results in
  relatively large errors. Cubic and quintic interpolation use stencils
  of four and six grid points, respectively.
  For cubic interpolation the first derivative is
  approximated with a centered finite difference formula. For quintic
  interpolation the same is done for the second derivative.
  Quintic interpolation performs significantly better than cubic interpolation,
  except in the boundary cells where the errors are comparable. In the boundary
  cells the effects of extrapolation dominate the error.

  \image html PolynomialInterpolationSineError_D0.jpg "Interpolation using only function values."
  \image latex PolynomialInterpolationSineError_D0.pdf "Interpolation using only function values."

  \par
  Next we examine cubic and quintic methods that use first derivative
  information. We see below that quintic interpolation, which uses four
  function values and two derivatives in each cell, is significantly more
  accurate than cubic interpolation, which uses two function values and
  two derivatives. Although it is a subtle effect in the plot below,
  we can see that the error in the quintic interpolant is largest
  in the boundary cells. Here one of the function values must be extrapolated.

  \image html PolynomialInterpolationSineError_D1.jpg "Interpolation using first derivative information."
  \image latex PolynomialInterpolationSineError_D1.pdf "Interpolation using first derivative information."

  \par
  Finally we compare two variants of quintic interpolation. One uses first
  derivative information (degree 1) and the other additionally uses second
  derivative information (degree 2). As we see below using the second
  derivative allows a more compact stencil and leads to significantly
  better accuracy. Note that the difference is greatest in the boundary cells
  where the degree 1 method must use extrapolation for one grid value.

  \image html PolynomialInterpolationSineError_O5.jpg "Quintic interpolation."
  \image latex PolynomialInterpolationSineError_O5.pdf "Quintic interpolation."

  \par
  In conclusion, if derivatives are available when sampling a function to
  be interpolated, one should use them. Interpolants that use first derivative
  information are more accurate than those that use only function values.
  Interpolants that use second derivatives are more accurate still.
  Regardless of what information is available, quintic interpolation is
  the most accurate. Of course accuracy may not be the sole concern. Next
  we will consider the computational cost.


  \par Performance.
  Below is a table of execution times for interpolation on grids of
  various sizes. The table is accessed in a random pattern. The results of
  the interpolations are accumulated so that the optimizer does not skip
  interpolation operations. The execution times are given in nanoseconds.

  \htmlinclude PolynomialInterpolationUsingCoefficients.txt

  \par
  In a loop that accumulates the result of a multiplication, each
  iteration takes about 3.2 nanoseconds. Thus we see that linear
  interpolation is cheap. Cubic interpolation is about 40% more expensive.
  Quintic interpolation is a bit more than twice as expensive as linear
  interpolation.
  Adding the calculation of the derivative(s)
  adds about 50% to the execution time. Because of the random access pattern,
  the performance degrades significantly when the grid size reaches 1,000,000.
  At that point the cache misses dominate the computational cost.
*/

//! Base class for polynomial interpolation on a 1-D regular grid.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1, 3, or 5.
*/
template<typename _T, std::size_t _Order>
class PolynomialInterpolationUsingCoefficientsBase :
  // The argument type is \c double and the return type is the
  // grid value type.
  public std::unary_function<double, _T>
{
  //
  // Types.
  //
private:

  //! The functor type.
  typedef std::unary_function<double, _T> Functor;

protected:

  //! The polynomial coefficients.
  typedef std::array < _T, _Order + 1 > Coefficients;

public:

  //! The argument type is \c double.
  typedef typename Functor::argument_type argument_type;
  //! The result type is the value type for the grid.
  typedef typename Functor::result_type result_type;
  //! The value type.
  typedef _T Value;

  //
  // Data.
  //
protected:

  //! The %array of coefficients for each cell.
  Coefficients* _coefficients;
  //! The number of cells.
  std::size_t _numberOfCells;
  //! The lower corner of the Cartesian domain spanned by the grid.
  double _lowerCorner;
  //! The inverse of the cell width.
  double _inverseWidth;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  //! @{
protected:

  //! Construct from the grid size and the Cartesian domain.
  /*!
    \param gridSize The grid size.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
    \note You must initialize this functor with the address of a grid before
    using it.
  */
  PolynomialInterpolationUsingCoefficientsBase(std::size_t gridSize,
      double lower, double upper);

  //! Deep copy. The grids must be the same size.
  void
  copy(const PolynomialInterpolationUsingCoefficientsBase& other);

  //! Copy the cell coefficients. The grids must be the same size.
  void
  copyCoefficients(const PolynomialInterpolationUsingCoefficientsBase& other);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  //! Return the number of interpolation cells.
  std::size_t getNumberOfCells() const
  {
    return _numberOfCells;
  }

  //! Return the number of grid points, which is one more than the number of cells.
  std::size_t getNumberOfGridPoints() const
  {
    return _numberOfCells + 1;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Mathematical operations.
  //! @{

  //! Add a constant to the interpolating function.
  /*!
    c<sub>0</sub> + c<sub>1</sub> x + c<sub>2</sub> x<sup>2</sup> + ... + v
    = (c<sub>0</sub> + v) + c<sub>1</sub> x + c<sub>2</sub> x<sup>2</sup> + ...
  */
  PolynomialInterpolationUsingCoefficientsBase&
  operator+=(const _T v)
  {
    for (std::size_t i = 0; i != _numberOfCells; ++i) {
      _coefficients[i][0] += v;
    }
    return *this;
  }

  //! Multiply the interpolating function by a constant.
  /*!
    (c<sub>0</sub> + c<sub>1</sub> x + c<sub>2</sub> x<sup>2</sup> + ...) v
    = c<sub>0</sub> v + c<sub>1</sub> v x + c<sub>2</sub> v x<sup>2</sup> + ...
  */
  PolynomialInterpolationUsingCoefficientsBase&
  operator*=(const _T v)
  {
    for (std::size_t i = 0; i != _numberOfCells; ++i) {
      _coefficients[i] *= v;
    }
    return *this;
  }

  //! Add an interpolating function to this interpolating function.
  /*!
    a<sub>0</sub> + a<sub>1</sub> x + ... + b<sub>0</sub> + b<sub>1</sub> x + ...
    = (a<sub>0</sub> + b<sub>0</sub>) + (a<sub>1</sub> + b<sub>1</sub>) x + ...
  */
  PolynomialInterpolationUsingCoefficientsBase&
  operator+=(const PolynomialInterpolationUsingCoefficientsBase& p)
  {
    assert(_numberOfCells == p._numberOfCells);
    for (std::size_t i = 0; i != _numberOfCells; ++i) {
      _coefficients[i] += p._coefficients[i];
    }
    return *this;
  }

  void
  zero()
  {
    for (std::size_t i = 0; i != _numberOfCells; ++i) {
      for (std::size_t o = 0; o != _Order + 1; ++o) {
        _coefficients[i][o] = 0.0;
      }
    }
  }

  void
  accumulateScaledInterpolator
  (const PolynomialInterpolationUsingCoefficientsBase& p, const _T v)
  {
    assert(_numberOfCells == p._numberOfCells);
    for (std::size_t i = 0; i != _numberOfCells; ++i) {
      _coefficients[i] += (p._coefficients[i] * v);
    }
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Interpolate the field at the specified point.
  result_type
  operator()(argument_type x) const;

  //! Interpolate the function and its derivative.
  result_type
  operator()(argument_type x, Value* derivative) const;

  //! Interpolate the function and its first and second derivatives.
  result_type
  operator()(argument_type x, Value* firstDerivative, Value* secondDerivative)
  const;

  //! Interpolate the field at the specified point.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x) const;

  //! Interpolate the function and its derivative.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x, Value* derivative) const;

  //! Interpolate the function and its first and second derivatives.
  /*! The interpolation order must be specified explicitly and must be
    no greater than the default order used to construct this class. */
  template<std::size_t _InterpolationOrder>
  result_type
  interpolate(argument_type x, Value* firstDerivative, Value* secondDerivative)
  const;

  //! Use the grid values to set the polynomial coefficient values.
  /*!
    \param f The first in the range of function values.
  */
  template<typename _ForwardIterator>
  void
  setGridValues(_ForwardIterator f);

  //! Use the grid values to set the polynomial coefficient values.
  /*!
    \param f The first in the range of function values.
    \param df The first in the range of derivative values.
  */
  template<typename _ForwardIterator>
  void
  setGridValues(_ForwardIterator f, _ForwardIterator df);

  //! Use the grid values to set the polynomial coefficient values.
  /*!
    \param f The first in the range of function values.
    \param df The first in the range of first derivative values.
    \param ddf The first in the range of second derivative values.
  */
  template<typename _ForwardIterator>
  void
  setGridValues(_ForwardIterator f, _ForwardIterator df,
                _ForwardIterator ddf);

private:

  //! Linear interpolation.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              std::integral_constant<std::size_t, 1> /*Order*/) const;

  //! Cubic interpolation.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              std::integral_constant<std::size_t, 3> /*Order*/) const;

  //! Cubic interpolation with derivative.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              Value* derivative,
              std::integral_constant<std::size_t, 3> /*Order*/) const;

  //! Quintic interpolation.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              std::integral_constant<std::size_t, 5> /*Order*/) const;

  //! Quintic interpolation with first derivative.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              Value* derivative,
              std::integral_constant<std::size_t, 5> /*Order*/) const;

  //! Quintic interpolation with first and second derivatives.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              Value* firstDerivative, Value* secondDerivative) const;

  //
  // Function.
  //

  void
  setCoefficients(const Value* f,
                  std::integral_constant<std::size_t, 1> /*Order*/);

  void
  setCoefficients(const Value* f,
                  std::integral_constant<std::size_t, 3> /*Order*/);

  void
  setCoefficients(const Value* f,
                  std::integral_constant<std::size_t, 5> /*Order*/);

  //
  // Function and first derivative.
  //

  void
  setCoefficients(const Value* f, const Value* /*df*/,
                  std::integral_constant<std::size_t, 1> /*Order*/);

  void
  setCoefficients(const Value* f, const Value* df,
                  std::integral_constant<std::size_t, 3> /*Order*/);

  void
  setCoefficients(const Value* f, const Value* df,
                  std::integral_constant<std::size_t, 5> /*Order*/);

  //
  // Function, first derivative, and second derivative.
  //

  void
  setCoefficients(const Value* f, const Value* /*df*/, const Value* /*ddf*/,
                  std::integral_constant<std::size_t, 1> /*Order*/);

  void
  setCoefficients(const Value* f, const Value* df, const Value* /*ddf*/,
                  std::integral_constant<std::size_t, 3> /*Order*/);

  void
  setCoefficients(const Value* f, const Value* df, const Value* ddf,
                  std::integral_constant<std::size_t, 5> /*Order*/);

  void
  checkErrors() const;

  //! @}
};


} // namespace numerical
}

#define __numerical_interpolation_PolynomialInterpolationUsingCoefficientsBase_ipp__
#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficientsBase.ipp"
#undef __numerical_interpolation_PolynomialInterpolationUsingCoefficientsBase_ipp__

#endif

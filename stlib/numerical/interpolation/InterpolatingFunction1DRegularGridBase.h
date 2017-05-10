// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunction1DRegularGridBase.h
  \brief Functor for polynomial interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunction1DRegularGridBase_h__)
#define __numerical_interpolation_InterpolatingFunction1DRegularGridBase_h__

#include <boost/config.hpp>

#include <functional>
#include <iterator>

#include <cassert>

namespace stlib
{
namespace numerical
{

/*!
  \page interpolation_InterpolatingFunction1DRegularGrid Functors for polynomial interpolation on a 1-D regular grid.

  \par Overview.
  The functors numerical::InterpolatingFunction1DRegularGrid and
  numerical::InterpolatingFunction1DRegularGridReference
  both support 1<sup>st</sup> order (linear), and 3<sup>rd</sup> order
  (cubic) interpolation on regular (uniformly spaced) grids.
  The difference between these two classes is that
  the former allocates memory for the grid whereas the latter
  references externally allocated memory.
  The domain associated with the grid exactly bounds the grid points.

  \par Multiple grids.
  If one is performing interpolation on multiple grids that have the same
  size and Cartesian domain then it is probably convenient to use the
  numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids functor. This class
  manages the memory for a multi-dimensional %array of grids.
  (The %array dimension is a template parameter.)
  One can access and manipulate the grids.
  One selects the desired grid before interpolating.
  Using this class is more efficient that allocating a multi-dimension %array
  of either numerical::InterpolatingFunction1DRegularGrid or
  numerical::InterpolatingFunction1DRegularGridReference because it does not
  store redundant information about the grids and because all of the data
  for the grids is stored in one contiguous %array.

  \par Using numerical::InterpolatingFunction1DRegularGrid.
  In the code example below we construct a linear interpolating function.
  To construct a numerical::InterpolatingFunction1DRegularGrid one specifies
  the function values sampled on a grid and the lower and upper bounds of
  the domain spanned by the grid.

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
  typedef numerical::InterpolatingFunction1DRegularGrid<double, 1> F;
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
  F f(grid.begin(), grid.end(), Lower, Upper);
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
  f.setGridValues(grid.begin(), grid.end());
  \endcode

  \par
  To use cubic interpolation one specifies an interpolation order of
  three as a template parameter when constructing the functor.
  The constructor arguments are the same as for linear interpolation.
  One may either evaluate the function or the function and its derivative.

  \code
  typedef numerical::InterpolatingFunction1DRegularGrid<double, 3> F;
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
  F f(grid.begin(), grid.end(), Lower, Upper);
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

  \par Referencing grid memory.
  To construct a
  numerical::InterpolatingFunction1DRegularGridReference one specifies
  a pointer to the grid data, the grid size and the Cartesian domain
  in the constructor. Below we show the differences for linear interpolation.

  \code
  typedef numerical::InterpolatingFunction1DRegularGridReference<double, 1> F;
  ...
  F f(&grid[0], grid.size(), Lower, Upper);
  ...
  \endcode

  \par
  The situation is more complicated when using cubic interpolation. This is
  because the formula uses four function values. For the sake of efficiency
  we store an extra guard point at each end of the grid. This avoids the
  branches to check if we are interpolating in the first or last cell. However,
  this complicates the allocation of the grid when using
  numerical::InterpolatingFunction1DRegularGridReference for cubic
  interpolation. Note in the example code below we do not set the values at
  the guard points, this will be done automatically in the constructor.

  \code
  typedef numerical::InterpolatingFunction1DRegularGridReference<double, 3> F;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  const std::size_t gridSize = 11;
  std::vector<double> gridData(gridSize + 2);
  double* grid = &gridData[1];
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (gridSize - 1);
  for (std::size_t i = 0; i != gridSize; ++i) {
     grid[i] = std::exp(Dx * i);
  }
  // Construct the interpolating function.
  F f(grid, gridSize, Lower, Upper);
  // Check that the function has the correct values at the grid points.
  // Note that we cannot evaluate the interpolating function at values
  // greater than or equal to the last grid point.
  const std::size_t numberOfCells = gridSize - 1;
  for (std::size_t i = 0; i != numberOfCells; ++i) {
     assert(numerical::areEqual(f(Dx * i), grid[i]));
  }
  // Change the interpolating function to sample the function f(x) = x^2.
  for (std::size_t i = 0; i != gridSize; ++i) {
     grid[i] = (Dx * i) * (Dx * i);
  }
  f.setGridValues(grid, grid + gridSize);
  \endcode

  \par Using an %array of grids.
  To construct a
  numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids one specifies
  the grid size, the Cartesian domain, and the %array extents in the
  constructor. One selects a grid with setGrid(). One may then set the
  grid values with setGridValues() or perform interpolation with the
  standard interface.

  \code
  // Linear interpolation. 2-D array of grids.
  const std::size_t Dimension = 2;
  typedef numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids<double, 1, Dimension> F;
  typedef F::SizeList SizeList;
  typedef F::IndexList IndexList;
  typedef container::MultiIndexRange<Dimension> ArrayRange;
  typedef container::MultiIndexRangeIterator<Dimension> ArrayIterator;
  // Make a grid to sample the exponential function on the domain [0..1).
  // Use a grid spacing of 0.1.
  std::size_t gridSize = 11;
  const double Lower = 0;
  const double Upper = 1;
  const double Dx = (Upper - Lower) / (gridSize - 1);
  const SizeList arrayExtents = {{10, 20}};
  // Construct the interpolating function.
  F f(gridSize, Lower, Upper, arrayExtents);
  // Set values of each of the grids.
  const ArrayRange arrayRange(arrayExtents);
  const ArrayIterator arrayBegin = ArrayIterator::begin(arrayRange);
  const ArrayIterator arrayEnd = ArrayIterator::end(arrayRange);
  std::vector<double> grid(gridSize);
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
     f.setGridValues(grid.begin(), grid.end());
  }
  // Check that the function has the correct values at the grid points.
  const std::size_t numberOfCells = grid.size() - 1;
  for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
     f.setGrid(*a);
     const double offset = sum(*a);
     for (std::size_t i = 0; i != numberOfCells; ++i) {
        assert(numerical::areEqual(f(Dx * i), offset + std::exp(Dx * i)));
     }
  }
  \endcode

  \par
  The interface for cubic interpolation is the same (except for the addition
  of the function call operator that returns the interpolated value and computes
  the derivative). The guard cells are allocated and initialized automatically.

  \par Feature comparison of the functors.
  As mentioned above the difference between
  numerical::InterpolatingFunction1DRegularGrid (RG) and
  numerical::InterpolatingFunction1DRegularGridReference (RGR)
  is that the former allocates memory for the grid whereas the latter
  references externally allocated memory. Both have copy constructors,
  and assignment operators. Note however that to use assignment with
  RGR's the grids must be the same size. The RG class is easier to
  construct, because one does not need to allocate grid memory
  (and pad the grid with guard cells for cubic interpolation).

  \par
  If one has many grids with the same size and Cartesian domain and
  if the grids can be logically arranged in a multi-dimensional array then
  it is best to use the
  numerical::InterpolatingFunctionMultiArrayOf1DRegularGrids class. This class
  efficiently manages memory for the grids and allows the user to
  manipulate them.

  \par Performance.
  Below is a table of execution times for interpolation on grids of
  various sizes. The table is accessed in a random pattern. The results of
  the interpolations are accumulated so that the optimizer does not skip
  interpolation operations. The execution
  times for linear interpolation, cubic interpolation, and cubic interpolation
  with derivatives are given in nanoseconds.

  \htmlinclude InterpolatingFunction1DRegularGrid.txt

  \par
  In a loop that accumulates the result of a multiplication, each
  iteration takes about 3.2 nanoseconds. Thus we see that linear
  interpolation is cheap. Cubic interpolation takes about four times as
  long as linear interpolation. Adding the calculation of the derivative
  adds another 20% to the execution time. Because of the random access pattern,
  the performance degrades significantly when the grid size reaches 1,000,000.
  At that point the cache misses dominate the computational cost.
*/

//! Functor for linear interpolation on a 1-D regular grid.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1 or 3.
*/
template<typename _T, std::size_t _Order>
class InterpolatingFunction1DRegularGridBase :
  // The argument type is \c double and the return type is the
  // grid value type.
  public std::unary_function<double, _T>
{
  //
  // Constants.
  //
protected:

  //! The number of ghost points.
  BOOST_STATIC_CONSTEXPR std::size_t GhostWidth = _Order / 2;

  //
  // Types.
  //
private:

  //! The functor type.
  typedef std::unary_function<double, _T> Functor;

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

  //! The %array of function values.
  Value* _grid;
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
  InterpolatingFunction1DRegularGridBase(std::size_t gridSize,
                                         double lower, double upper) :
    _grid(0),
    // The number of cells is one less than the number of grid elements.
    _numberOfCells(gridSize - 1),
    _lowerCorner(lower),
    _inverseWidth(_numberOfCells / (upper - lower))
  {
  }

  //! Deep copy. The grids must be the same size.
  void
  copy(const InterpolatingFunction1DRegularGridBase& other)
  {
    assert(this != &other);
    copyGrid(other);
    // No need to copy _numberOfCells.
    _lowerCorner = other._lowerCorner;
    _inverseWidth = other._inverseWidth;
  }

  //! Copy the grid values. The grids must be the same size.
  void
  copyGrid(const InterpolatingFunction1DRegularGridBase& other)
  {
    assert(other._numberOfCells == _numberOfCells);
    // Copy the grid values and other member variables.
    std::copy(other._grid - GhostWidth,
              other._grid + _numberOfCells + 1 + GhostWidth,
              _grid - GhostWidth);
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Interpolate the field at the specified point.
  result_type
  operator()(argument_type x) const
  {
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

  //! Use cubic interpolation to evaluate the function and its derivative.
  result_type
  operator()(argument_type x, Value* derivative) const
  {
    static_assert(_Order == 3, "Not supported.");
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

    const Value x2 = x * x;
    const Value x3 = x2 * x;
    const Value d[] = {0.5 * (_grid[i + 1] - _grid[i - 1]),
                       0.5 * (_grid[i + 2] - _grid[i])
                      };
    *derivative = (_grid[i] - _grid[i + 1]) * 6 * (x2 - x)
                  + d[0] * (3 * x2 - 4 * x + 1)
                  + d[1] * (3 * x2 - 2 * x);
    // Scale to go from logical coordinates to physical coordinates.
    *derivative *= _inverseWidth;
    return _grid[i] * (2 * x3 - 3 * x2 + 1)
           + _grid[i + 1] * (-2 * x3 + 3 * x2)
           + d[0] * (x3 - 2 * x2 + x)
           + d[1] * (x3 - x2);
  }

  //! Set the grid values.
  /*!
    \param begin The first grid value.
    \param end One past the last grid value.
  */
  template<typename _ForwardIterator>
  void
  setGridValues(_ForwardIterator begin, _ForwardIterator end)
  {
    assert(std::size_t(std::distance(begin, end)) == _numberOfCells + 1);
    std::copy(begin, end, _grid);
    setGhost();
  }

protected:

  //! Set the ghost values if we are using cubic interpolation.
  void
  setGhost()
  {
    if (_Order == 3) {
      assert(_numberOfCells >= 1);
      _grid[-1] = 2 * _grid[0] - _grid[1];
      _grid[_numberOfCells + 1] = 2 * _grid[_numberOfCells]
                                  - _grid[_numberOfCells - 1];
    }
  }

private:

  //! Linear interpolation.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              std::integral_constant<std::size_t, 1> /*Order*/) const
  {
    return _grid[i] * (1. - x) + _grid[i + 1] * x;
  }

  //! Cubic interpolation.
  result_type
  interpolate(const argument_type x, const std::size_t i,
              std::integral_constant<std::size_t, 3> /*Order*/) const
  {
    const Value x2 = x * x;
    const Value x3 = x2 * x;
    return _grid[i] * (2 * x3 - 3 * x2 + 1)
           + _grid[i + 1] * (-2 * x3 + 3 * x2)
           + 0.5 * (_grid[i + 1] - _grid[i - 1]) * (x3 - 2 * x2 + x)
           + 0.5 * (_grid[i + 2] - _grid[i]) * (x3 - x2);
  }

  //! @}
};

} // namespace numerical
}

#endif

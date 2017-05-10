// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunctionRegularGridBase.h
  \brief Functor for interpolation on a regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunctionRegularGridBase_h__)
#define __numerical_interpolation_InterpolatingFunctionRegularGridBase_h__

#include "stlib/container/EquilateralArray.h"
#include "stlib/container/MultiArray.h"
#include "stlib/geom/kernel/BBox.h"

#include <functional>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

/*!
  \page interpolation_InterpolatingFunctionRegularGrid Functors for interpolation on a regular grid.

  \par Overview.
  The functors numerical::InterpolatingFunctionRegularGrid and
  numerical::InterpolatingFunctionRegularGridReference
  both support 0<sup>th</sup> order
  (constant), 1<sup>st</sup> order (linear), and 3<sup>rd</sup> order
  (cubic) interpolation on regular (uniformly spaced in each
  dimension) grids. The difference between these two classes is that
  the former allocates memory for the grid whereas the latter
  references externally allocated memory.
  The grid that holds function values may be either plain or periodic. For
  a plain grid the domain associated with the grid exactly bounds the
  grid points. This shown below.

  \image html interpolation/InterpolationPlain.png "The grid points and associated domain for a 2-D plain grid."

  \par Multiple grids.
  If one is performing interpolation on multiple grids that have the same
  extents and Cartesian domain then it is probably convenient to use the
  numerical::InterpolatingFunctionMultiArrayOfRegularGrids functor. This class
  manages the memory for a multi-dimensional %array of grids. (Both the grid
  dimension and the %array dimension are template parameters.)
  One can access and manipulate the grids as
  multi-dimensional arrays. One selects the desired grid before interpolating.
  Using this class is more efficient that allocating a multi-dimension %array
  of either numerical::InterpolatingFunctionRegularGrid or
  numerical::InterpolatingFunctionRegularGridReference because it does not
  store redundant information about the grids and because all of the data
  for the grids is stored in one contiguous %array.

  \par Using numerical::InterpolatingFunctionRegularGrid.
  In the code example below we construct a 2-D, linear interpolating function
  for the case of a plain grid. To construct a
  numerical::InterpolatingFunctionRegularGrid one specifies the function values
  sampled on a grid and the domain spanned by the grid.

  \code
  const std::size_t Dimension = 2;
  const std::size_t DefaultOrder = 1;
  typedef numerical::InterpolatingFunctionRegularGrid<double, Dimension, DefaultOrder>
     InterpolatingFunction;
  typedef InterpolatingFunction::Point Point;
  typedef InterpolatingFunction::BBox BBox;
  typedef InterpolatingFunction::Grid Grid;
  typedef Grid::SizeList SizeList;
  typedef container::MultiIndexRangeIterator<Dimension> Iterator;

  // Make a 5x7 2-D array.
  const SizeList extents = {{5, 7}};
  Grid grid(extents);
  // The Cartesian domain is (0..2)x(0..3).
  const Point lower = {{0, 0}};
  const Point upper = {{2, 3}};
  const Point dx = (upper - lower) / (extents - 1);
  // Set the array values to the sum of the point coordinates.
  const Iterator end = Iterator::end(grid.range());
  Point x;
  for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
     x = lower + *i * dx;
     grid(*i) = sum(x);
  }
  // Construct the interpolating function.
  const BBox domain(lower, upper);
  InterpolatingFunction f(grid, domain);
  // Check that the function has the correct values at the grid points.
  for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
     x = lower + *i * dx;
     assert(numerical::areEqual(f(x), sum(x)));
  }
  \endcode

  \par Referencing grid memory.
  To construct a
  numerical::InterpolatingFunctionRegularGridReference one specifies
  the grid extents and the Cartesian domain in the constructor. One then
  uses the setData() member function to reference the grid. Below are
  the differences for using this class.

  \code
  ...
  typedef numerical::InterpolatingFunctionRegularGridReference<double, Dimension, DefaultOrder>
     InterpolatingFunction;
  ...
  InterpolatingFunction f(extents, domain);
  f.setData(grid.begin());
  ...
  \endcode

  \par Using an %array of grids.
  To construct a
  numerical::InterpolatingFunctionMultiArrayOfRegularGrids one specifies
  the grid extents, the Cartesian domain, and the %array extents in the
  constructor. One uses the grid() member functions to access and manipulate
  the grids. To perform interpolation, one first selects a grid with
  setGrid() and then uses the standard interpolation interface.

  \code
  // A 3-D array of 2-D grids.
  const std::size_t GridDimension = 2;
  const std::size_t ArrayDimension = 3;
  const std::size_t DefaultOrder = 1;
  typedef numerical::InterpolatingFunctionMultiArrayOfRegularGrids
     <double, GridDimension, ArrayDimension, DefaultOrder>
     InterpolatingFunction;
  typedef InterpolatingFunction::ArraySizeList ArraySizeList;
  typedef InterpolatingFunction::GridSizeList GridSizeList;
  typedef InterpolatingFunction::GridRef GridRef;
  typedef InterpolatingFunction::Point Point;
  typedef InterpolatingFunction::BBox BBox;
  typedef container::MultiIndexRange<ArrayDimension> ArrayRange;
  typedef container::MultiIndexRangeIterator<ArrayDimension> ArrayIterator;
  typedef container::MultiIndexRange<GridDimension> GridRange;
  typedef container::MultiIndexRangeIterator<GridDimension> GridIterator;

  const GridSizeList gridExtents = {{2, 3}};
  // The Cartesian domain is (0..2)x(0..3).
  const Point lower = {{0, 0}};
  const Point upper = {{2, 3}};
  const Point dx = (upper - lower) / (gridExtents - 1);
  const BBox domain(lower, upper);
  const ArraySizeList arrayExtents = {{5, 7, 11}};
  InterpolatingFunction f(gridExtents, domain, arrayExtents);

  // Useful for iterating over the grids.
  const ArrayRange arrayRange(arrayExtents);
  const ArrayIterator arrayBegin = ArrayIterator::begin(arrayRange);
  const ArrayIterator arrayEnd = ArrayIterator::end(arrayRange);
  const GridRange gridRange(gridExtents);
  const GridIterator gridBegin = GridIterator::begin(gridRange);
  const GridIterator gridEnd = GridIterator::end(gridRange);

  // Set the array values to the sum of the point coordinates plus the
  // sum of the grid indices.
  Point x;
  // Loop over the grids in the array.
  for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
     GridRef grid = f.grid(*a);
     const double offset = sum(*a);
     // Loop over the elements in the grid.
     for (GridIterator g = gridBegin; g != gridEnd; ++g) {
        x = lower + *g * dx;
        grid(*g) = sum(x) + offset;
     }
  }

  // Check that the function has the correct values at the grid points.
  // Loop over the grids in the array.
  for (ArrayIterator a = arrayBegin; a != arrayEnd; ++a) {
     // Select the grid.
     f.setGrid(*a);
     const double offset = sum(*a);
     // Loop over the elements in the grid.
     for (GridIterator g = gridBegin; g != gridEnd; ++g) {
        x = lower + *g * dx;
        assert(numerical::areEqual(f(x), sum(x) + offset));
     }
  }
  \endcode

  \par Periodic grids.
  Let dx = (dx<sub>0</sub>, ..., dx<sub>N-1</sub>) be the grid spacings.
  For periodic grids the associated domain exactly bounds the grid at the
  lower bounds and is dx beyond the grid points at the upper bounds.
  For example, consider 1-D grid that samples function values in the
  domain [0...360) at 60 intervals. The grid would sample
  the function at the points 0, 60, 120, 180, 240,
  and 300. A 2-D periodic grid is depicted below.

  \image html interpolation/InterpolationPeriodic.png "The grid points and associated domain for a 2-D periodic grid."

  \par
  Below we construct an interpolating function for the 2-D function
  \f$\cos(x^\circ) \sin(y^\circ)\f$.

  \code
  const std::size_t Dimension = 2;
  const std::size_t DefaultOrder = 1;
  typedef numerical::InterpolatingFunctionRegularGrid<double, Dimension,
  DefaultOrder, true> InterpolatingFunction;
  typedef InterpolatingFunction::Point Point;
  typedef InterpolatingFunction::BBox BBox;
  typedef InterpolatingFunction::Grid Grid;
  typedef Grid::SizeList SizeList;
  typedef Grid::Index Index;
  typedef container::MultiIndexRangeIterator<Dimension> Iterator;

  // Define a grid to sample the function at every 30 degrees.
  const SizeList extents = {{12, 12}};
  Grid grid(extents);
  const Point lower = {{0, 0}};
  const Point upper = {{360, 360}};
  const Point dx = (upper - lower) / extents;
  // Sample the function.
  const double Deg = numerical::Constants<double>::Degree();
  const Iterator end = Iterator::end(grid.range());
  Point x;
  for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
     x = lower + *i * dx;
     const double y = std::cos(x[0] * Deg) * std::sin(x[1] * Deg);
     grid(*i) = y;
  }
  // Construct an interpolating function.
  const BBox domain(lower, upper);
  InterpolatingFunction f(grid, domain);
  // Ensure that the function has the correct values at the grid points.
  for (Iterator i = Iterator::begin(grid.range()); i != end; ++i) {
     x = lower + *i * dx;
     const double y = std::cos(x[0] * Deg) * std::sin(x[1] * Deg);
     assert(numerical::areEqual(f(x), y));
  }
  \endcode

  \par
  In order to define the function for all points in R<sup>N</sup> a periodic
  grid is virtually duplicated. This
  is depicted for a 2-D grid below. The set of twelve grid points are repeated
  in each direction.

  \image html interpolation/InterpolationPeriodicExtension.png "The periodic extension of a 2-D grid."




  \par Constant interpolation.
  We will consider each interpolation order in turn, starting with
  0<sup>th</sup> order. Let \e f be a function sampled on an N-dimensional
  regular grid and let \e y be the interpolating function defined for
  \f$x \in \mathbf{R}^N\f$. For constant interpolation the value of the
  interpolating function is the value of \e f at the closest grid point.
  The domain associated with each grid point is shown below for a
  plain (non-periodic) grid in 2-D.

  \image html interpolation/Interpolation0Plain.png "A 2-D plain grid. The black box indicates the domain. Blue lines delimit the domain associated with each grid point."

  \par
  For a periodic grid the domain associated with each grid point extends half
  of the grid spacing around the point.

  \image html interpolation/Interpolation0Periodic.png "A 2-D periodic grid. The black box indicates the domain. Blue lines delimit the domain associated with each grid point."



  \par Linear interpolation.
  Next consider 1<sup>st</sup>-order interpolation. We use multi-linear
  interpolation, meaning that the interpolating function is a linear
  function in each coordinate. In 1-D this is called
  linear interpolation
  and has the functional form \f$y(\mathbf{x}) = c_0 + c_1 x_0\f$.
  In 2-D it is called
  <a href="http://en.wikipedia.org/wiki/Bilinear_interpolation">bilinear
  interpolation</a>
  and has the form
  \f$y(\mathbf{x}) = c_{00} + c_{10} x_0 + c_{01} x_1 + c_{11} x_0 x_1\f$.
  In 3-D the method is called
  <a href="http://en.wikipedia.org/wiki/Trilinear_interpolation">trilinear
  interpolation</a>. In an N-D grid, groups of 2<sup>N</sup> adjacent points
  form \e cells. For a point \e x in a cell \e c, the 2<sup>N</sup> function
  values defined at the grid points are used to perform multi-linear
  interpolation. For a plain grid, if a point lies outside the grid domain,
  the closest cell is used for extrapolation. This is indicated below.

  \image html interpolation/Interpolation1Plain.png "A 2-D plain grid. Blue lines delimit the domain associated with each cell."

  \par
  For a periodic grid, the grid points are virtually repeated so that the
  cells cover all of \f$\mathbf{R}^N\f$. Below we show the cells in
  grid domain.

  \image html interpolation/Interpolation1Periodic.png "A 2-D periodic grid. Blue lines delimit the domain associated with each cell."


  \par Cubic interpolation.
  In 1-D, 3<sup>rd</sup> order interpolation is performed with
  <a href="http://en.wikipedia.org/wiki/Cubic_interpolation">cubic %Hermite
  splines</a>. For 2-D grids, we use
  <a href="http://en.wikipedia.org/wiki/Bicubic_interpolation">bicubic
  interpolation</a>. As with linear interpolation, the cell containing
  the grid point is used to determine the interpolation coefficients.
  However, for cubic interpolation we also need the function derivatives
  evaluated at the grid points. For %Hermite interpolation in 1-D we use
  the values of the function and its derivative evaluated at the grid points
  of the surrounding cell. These four values determine the four coefficients
  in the polynomial. Note that we do not store the derivative values.
  They are computed as they are needed by using centered differencing. Thus
  %Hermite interpolation depends on four function values.

  \par
  Consider bicubic interpolation in 2-D. The interpolation coefficients are
  for a function \e f are determined by the values of \e f, \e f<sub>x</sub>,
  \e f<sub>y</sub>, and \e f<sub>xy</sub> evaluated at the four cell grid
  points. In order to compute the derivative values, we use centered
  differencing on the 4x4 stencil of function values that surround the
  cell. Thus 16 function values are transformed to 4 function values,
  8 first derivatives, and 4 cross derivatives. For periodic grids the
  function values are always defined on the 4<sup>N</sup> grid points of the
  stencil. However, on the boundary of plain grids there will be undefined
  values. In this case we use linear extrapolation using the function
  values at the cell grid points to define values for points outside
  of the grid.

  \par
  For cubic interpolation one can interpolate
  the gradient as well as the function values.
  (Interpolating the gradient is not supported for lower interpolation ordes.
  Although one could define the gradient for linear
  interpolation, it would not be continuous. Most applications that
  use the gradient require continuity.)
  Cubic interpolation is supported only in 1-D and 2-D spaces.
  Attempting to use unsupported functionality will result in a compilation
  error.


  \par Interpolation interface.
  For each of the interpolating functors there are a variety of ways to
  perform interpolation. The easiest method is to use the function call
  operator. This uses the default interpolation order specified as a template
  parameter.

  \code
  typedef numerical::InterpolatingFunctionRegularGrid<double, Dimension, DefaultOrder>
     InterpolatingFunction;
  ...
  InterpolatingFunction f(grid, domain);
  Point x;
  ...
  double value = f(x);
  ...
  Point gradient;
  value = f(x, &gradient);
  \endcode

  \par
  One may also specify the interpolation order as a template parameter to
  the interpolate() member function. This allows one to perform interpolation
  with any of the supported orders.

  \code
  InterpolatingFunction f(grid, domain);
  Point x;
  ...
  double constant = f.interpolate<0>(x);
  double linear = f.interpolate<1>(x);
  double cubic = f.interpolate<3>(x);
  ...
  Point gradient;
  constant = f.interpolate<0>(x, &gradient);
  linear = f.interpolate<1>(x, &gradient);
  cubic = f.interpolate<3>(x, &gradient);
  \endcode

  \par
  Additionally, one may specify the interpolation order as a function
  parameter. However, because of branching this is less efficient than
  the other interfaces. Thus one should use this interface only when the
  interpolation order is not known at compile-time.

  \code
  double constant = f.interpolate(0, x);
  double linear = f.interpolate(1, x);
  double cubic = f.interpolate(3, x);
  ...
  constant = f.interpolate(0, x, &gradient);
  linear = f.interpolate(1, x, &gradient);
  cubic = f.interpolate(3, x, &gradient);
  \endcode


  \par Feature comparison of the functors.
  As mentioned above the difference between
  numerical::InterpolatingFunctionRegularGrid (RG) and
  numerical::InterpolatingFunctionRegularGridReference (RGR)
  is that the former allocates memory for the grid whereas the latter
  references externally allocated memory. Both have copy constructors,
  but only RG has as an assignment
  operator. One might have a \c std::vector of
  RG, however one cannot do the
  same for the other class. This does not limit the limit the functionality
  of RGR. One may store
  a container of grids and use a single RGR, calling setData() to switch
  between the grids.

  \par
  If one has many grids with the same extents and Cartesian domain and
  the grids can be logically arranged in a multi-dimensional array then
  it is best to use the
  numerical::InterpolatingFunctionMultiArrayOfRegularGrids class. This class
  efficiently manages memory for the grids and allows the user to access or
  manipulate them.

  \par Performance.
  Below is a table of execution times for interpolation on grids with a
  total of 4096 elements. The table is accessed in a random pattern.
  The results of
  the interpolations are accumulated so that the optimizer does not skip
  interpolation operations. The execution
  times for linear interpolation, cubic interpolation, and cubic interpolation
  with derivatives are given in nanoseconds.

  \htmlinclude InterpolatingFunctionRegularGrid.txt

  \par
  Note that the execution times for 1-D grids are much longer than for
  the \ref interpolation_InterpolatingFunction1DRegularGrid "functors"
  that have been specialized for 1-D grids. This performance disparity is
  due to several factors:
  - The N-D functors check whether the argument is within the grid domain.
  If it is not, they perform extrapolation using the closest grid cell.
  The 1-D specialization does perform this check.
  - The 1-D specialization stores guard points when performing cubic
  interpolation. The N-D functors do not and must extrapolate to fill
  temporary guard points when using boundary cells.
  - The N-D functors copy data from the grid to fixed-size arrays before
  performing the interpolation. While this greatly simplifies the program
  logic, it does add some computational overhead.
*/

//! Functor for interpolation on a regular grid.
/*!
  \param _T The value type.
  \param _Dimension The space dimension.
  \param _DefaultOrder The default interpolation order.
  \param _IsPeriodic True if the grid data is periodic. By default this is
  false.
*/
template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic = false>
class InterpolatingFunctionRegularGridBase :
  // The argument type is a Cartesian point and the return type is the
  // value type.
  public std::unary_function<std::array<double, _Dimension>, _T>
{
  //
  // Types.
  //
private:

  //! The functor type.
  typedef std::unary_function<std::array<double, _Dimension>, _T>
  Functor;

public:
  //! The argument type is a Cartesian point.
  typedef typename Functor::argument_type argument_type;
  //! The result type is the field.
  typedef typename Functor::result_type result_type;
  //! The value type.
  typedef _T Value;
  //! A Cartesian point.
  typedef std::array<double, _Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<double, _Dimension> BBox;
  //! The %array of function values.
  typedef container::MultiArrayConstRef<Value, _Dimension> GridConstRef;
  //! The (multi) size type.
  typedef typename GridConstRef::SizeList SizeList;
  //! The (multi) index type.
  typedef typename GridConstRef::IndexList IndexList;
  //! The single index type.
  typedef typename GridConstRef::Index Index;

  //
  // Data.
  //
private:
  //! The %array of function values.
  const GridConstRef* _gridPointer;
  //! The number of cells in each dimension.
  SizeList _cellExtents;
  //! The inverse of the number of cells in each dimension.
  /*! This quantity is stored to avoid division. */
  Point _inverseCellExtents;
  //! The lower corner of the Cartesian domain spanned by the grid.
  Point _lowerCorner;
  //! The inverse of the cell widths.
  Point _inverseWidths;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Default constructor. Invalid state.
  /*! \note You must initialize this functor before using it. */
  InterpolatingFunctionRegularGridBase();

  //! Construct from the %array of function values and the Cartesian domain.
  /*!
    \param extents The %array extents.
    \param domain The Cartesian domain.
    \note You must initialize this functor with the address of a grid before
    using it.
  */
  InterpolatingFunctionRegularGridBase(const SizeList& extents,
                                       const BBox& domain);

  // The default copy constructor, assignment operator, and destructor are
  // fine.

protected:

  //! Set the grid extents and domain.
  /*!
    \param extents The %array extents.
    \param domain The Cartesian domain.
  */
  void
  initialize(const SizeList& extents, const BBox& domain);

  void
  initialize(const GridConstRef* grid)
  {
    _gridPointer = grid;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation with a statically specified order.
  //! @{
public:

  //! Interpolate the field at the specified point.
  template<std::size_t _Order>
  result_type
  interpolate(argument_type x) const;

  //! Interpolate the field at the specified point and calculate the gradient.
  template<std::size_t _Order>
  result_type
  interpolate(argument_type x,
              std::array<_T, _Dimension>* gradient) const;

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation with a dynamically specified order.
  //! @{
public:

  //! Interpolate the field at the specified point.
  /*!
    \note This is a little less efficient than using the versions with
    statically determined order.
  */
  result_type
  interpolate(std::size_t order, const argument_type& x) const;

  //! Interpolate the field at the specified point and calculate the gradient.
  /*!
    \note This is a little less efficient than using the versions with
    statically determined order.
  */
  result_type
  interpolate(std::size_t order, const argument_type& x,
              std::array<_T, _Dimension>* gradient) const;

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Interpolate the field at the specified point.
  result_type
  operator()(const argument_type& x) const
  {
    return interpolate<_DefaultOrder>(x);
  }

  //! Interpolate the field at the specified point and calculate the gradient.
  result_type
  operator()(const argument_type& x,
             std::array<_T, _Dimension>* gradient) const
  {
    return interpolate<_DefaultOrder>(x, gradient);
  }

  //! Calculate the closest grid point to the Cartesian location.
  void
  snapToGrid(const argument_type& x, IndexList* index) const
  {
    argument_type y = x;
    calculateScaledCoordinates(index, &y,
                               std::true_type() /*point-based*/);
  }

  //! @}
private:

  // A level of indirection for cell-based vs. point-based coordinates.
  // Zeroth-order interpolation is point-based. Higher order schemes are
  // cell-based.
  template<std::size_t _Order>
  void
  calculateScaledCoordinates(IndexList* cell, argument_type* x) const
  {
    calculateScaledCoordinates(cell, x,
                               std::integral_constant<bool, _Order == 0>());
  }

  // Calculate coordinates for first-order or higher interpolation.
  void
  calculateScaledCoordinates(IndexList* cell, argument_type* x,
                             std::false_type /*point-based*/) const;

  // Calculate coordinates for zeroth-order interpolation.
  void
  calculateScaledCoordinates(IndexList* cell, argument_type* x,
                             std::true_type /*point-based*/) const;
};

} // namespace numerical
}

#define __numerical_interpolation_InterpolatingFunctionRegularGridBase_ipp__
#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGridBase.ipp"
#undef __numerical_interpolation_InterpolatingFunctionRegularGridBase_ipp__

#endif

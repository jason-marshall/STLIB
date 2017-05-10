// -*- C++ -*-

/*!
  \file numerical/interpolation/QuinticInterpolation2D.h
  \brief Quintic interpolation on a regular 2-D grid.
*/

#if !defined(__numerical_interpolation_QuinticInterpolation2D_h__)
#define __numerical_interpolation_QuinticInterpolation2D_h__

#include "stlib/numerical/polynomial/Polynomial.h"

#include "stlib/container/MultiArray.h"
#include "stlib/container/EquilateralArray.h"
#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Function value, gradient, and Hessian.
template<typename _T>
struct ValueGradientHessian {
  //! The function value.
  _T f;
  //! df/dx
  _T fx;
  //! df/dy
  _T fy;
  //! d<sup>2</sup>f/dx<sup>2</sup>
  _T fxx;
  //! d<sup>2</sup>f/dxdx
  _T fxy;
  //! d<sup>2</sup>f/dy<sup>2</sup>
  _T fyy;
};


//! Write the components of \c x separated by spaces.
/*! \relates ValueGradientHessian */
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const ValueGradientHessian<_T>& x)
{
  return out << x.f << ' '
         << x.fx << ' '
         << x.fy << ' '
         << x.fxx << ' '
         << x.fxy << ' '
         << x.fyy;
}

//! Constants for QuinticInterpolation2D.
template<typename _T>
class QuinticInterpolation2DConstants
{
protected:
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = 2;

protected:

  //! An infinitessimal that is suitable for finite differencing.
  /*! The best accuracy that one can hope for is &epsilon;<sup>1/2</sup>.
    Since centered differencing is second order accurace we use an offset
    of &epsilon;<sup>1/4</sup>. */
  const _T _delta;
  // The multiplicative inverse of the delta.
  const _T _inverseDelta;
  //! The tuple (-Delta, 0, Delta).
  const std::array<_T, 3> _offsets;


public:
  // Use the default copy constructor and destructor.

  //! Default constructor.
  QuinticInterpolation2DConstants() :
    _delta(std::pow(std::numeric_limits<_T>::epsilon(), 0.25)),
    _inverseDelta(1 / _delta),
    _offsets(std::array<_T, 3>{{-_delta, 0, _delta}})
  {
  }

  //! Assignment operator.
  QuinticInterpolation2DConstants&
  operator=(const QuinticInterpolation2DConstants&)
  {
    // There is only BOOST_STATIC_CONSTEXPR and const data.
    return *this;
  }
};

//! Quintic interpolation on a regular 2-D grid.
/*!
  \param _T The number type.
  \param _IsPeriodic Whether the grid is periodic. The default value is false.

  \par
  This class performs quintic interpolation on 2-D data that is defined on a
  regular rectilinear grid. The interpolation returns the function
  value \e f as well as the first and second derivatives:
  <em>f<sub>x</sub></em>,
  <em>f<sub>y</sub></em>,
  <em>f<sub>xx</sub></em>,
  <em>f<sub>xy</sub></em>, and
  <em>f<sub>yy</sub></em>.
  The function value and its derivatives are represented with the
  ValueGradientHessian struct.

  \par Plain and periodic grids.
  The sampling grid may be either plain or periodic. For
  a plain grid the domain associated with the grid exactly bounds the
  grid points. This shown below.

  \image html interpolation/InterpolationPlain.png "The grid points and associated domain for a plain grid."

  \par
  Let d<em>x</em> = (d<em>x</em><sub>0</sub>, d<em>x</em><sub>1</sub>)
  be the grid spacings.
  For periodic grids the associated domain exactly bounds the grid at the
  lower bounds and is d<em>x</em> beyond the grid points at the upper bounds.
  For example, consider a grid that samples function values in the
  domain [0&deg;...360&deg;)&times;[0&deg;...360&deg;) at 60&deg;
  intervals. The grid would sample
  the function at 0&deg;, 60&deg;, 120&deg;, 180&deg;, 240&deg;,
  and 300&deg; in each coordinate. A periodic grid is depicted below.

  \image html interpolation/InterpolationPeriodic.png "The grid points and associated domain for a periodic grid."

  \par
  The second template parameter is a boolean value that specifies whether
  the grid is periodic. The default value is <code>false</code> (plain grid).
  Thus the following two statements construct plain grids.
  \code
  numerical::QuinticInterpolation2D<double, false> f;
  numerical::QuinticInterpolation2D<double> g;
  \endcode
  Specify <code>true</code> to use interpolation on a periodic grid.
  \code
  numerical::QuinticInterpolation2D<double, true> f;
  \endcode

  \par Accuracy.
  As input, the function values are specified at the grid points. Because the
  derivatives are not supplied, they are approximated with finite difference
  schemes. At interior grid points centered difference schemes yield
  derivative values with <em>O(dx<sup>2</sup>)</em> errors, where
  <em>dx</em> is the grid spacing. (For periodic grids, all points are interior
  points.) At boundary grid points we use
  backward or forward difference schemes. There the first order derivatives
  have <em>O(dx<sup>2</sup>)</em> errors while the second order derivatives
  have <em>O(dx)</em> errors.

  \par
  An interpolation cell has four grid points at its corners.
  The errors for the interpolated values are:
  <em>O(dx<sup>3</sup>)</em> for the function,
  <em>O(dx<sup>2</sup>)</em> for the first derivatives, and
  <em>O(dx)</em> for the second derivatives.

  \par Usage.
  To construct an interpolation functor supply a grid of function values
  and the Cartesian domain of the grid. One may either use the constructor
  that takes these arguments or construct with the default constructor
  and use initialize(). The function call operator performs the interpolation.
  It takes a position and a pointer to a ValueGradientHessian as
  arguments. In the example below we construct an interpolating function
  and evaluate the interpolant.

  \code
  // Define types.
  typedef numerical::QuinticInterpolation2D<double> Interp;
  typedef Interp::ValueGradientHessian ValueGradientHessian;
  typedef Interp::Point Point;
  typedef Interp::BBox BBox;
  typedef Interp::ValueGrid ValueGrid;
  typedef ValueGrid::SizeList SizeList;
  typedef ValueGrid::IndexList IndexList;
  typedef ValueGrid::Index Index;

  // The grid has 100 x 100 cells. (101 x 101 grid points.)
  const SizeList Extents = {{101, 101}};
  const double Pi = numerical::Constants<double>::Pi();
  const Point Lo = {{0, 0}};
  const Point Hi = {{Pi, Pi}};
  const Point Delta = (Hi - Lo) / (Extents - 1);
  const BBox Domain(Lo, Hi);
  ValueGrid valueGrid(Extents);
  // Sample the function.
  IndexList i;
  Point x;
  for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
     for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        x = i * Delta;
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
     }
  }

  // Construct the interpolating function.
  Interp interp(valueGrid, Domain);

  // Evaluate the interpolant at (pi/3, pi/3).
  ValueGradientHessian y;
  x[0] = Pi / 3;
  x[1] = Pi / 3;
  interp(x, &y);

  // Check the accuracy of the interpolant.
  const double Dx = max(Delta);
  assert(std::abs(y.f - std::cos(x[0]) * std::sin(x[1])) < Dx * Dx * Dx);
  assert(std::abs(y.fx + std::sin(x[0]) * std::sin(x[1])) < Dx * Dx);
  assert(std::abs(y.fy - std::cos(x[0]) * std::cos(x[1])) < Dx * Dx);
  assert(std::abs(y.fxx + std::cos(x[0]) * std::sin(x[1])) < Dx);
  assert(std::abs(y.fxy + std::sin(x[0]) * std::cos(x[1])) < Dx);
  assert(std::abs(y.fyy + std::cos(x[0]) * std::sin(x[1])) < Dx);
  \endcode

  \par
  Next we consider the case of sampling a periodic function. The example
  below shows the differences in using a periodic grid.

  \code
  // Define types.
  typedef numerical::QuinticInterpolation2D<double, true> Periodic;
  typedef Periodic::ValueGradientHessian ValueGradientHessian;
  typedef Periodic::Point Point;
  typedef Periodic::BBox BBox;
  typedef Periodic::ValueGrid ValueGrid;
  typedef ValueGrid::SizeList SizeList;
  typedef ValueGrid::IndexList IndexList;
  typedef ValueGrid::Index Index;

  // The grid has 100 x 100 cells. (100 x 100 grid points.)
  const SizeList Extents = {{100, 100}};
  const double Pi = numerical::Constants<double>::Pi();
  const Point Lo = {{0, 0}};
  const Point Hi = {{2 * Pi, 2 * Pi}};
  const Point Delta = (Hi - Lo) / Extents;
  const BBox Domain(Lo, Hi);
  ValueGrid valueGrid(Extents);
  // Sample the function.
  IndexList i;
  Point x;
  for (i[0] = 0; i[0] != Index(valueGrid.extents()[0]); ++i[0]) {
     for (i[1] = 0; i[1] != Index(valueGrid.extents()[1]); ++i[1]) {
        x = i * Delta;
        valueGrid(i) = std::cos(x[0]) * std::sin(x[1]);
     }
  }

  // Construct the interpolating function.
  Periodic interp(valueGrid, Domain);
  \endcode

  \note This class was designed for the scenario that there is no derivative
  information for the sampled function. If it is used in a context in
  which this information is available, one would want to add a constructor
  that uses the derivatives. Analytical derivatives are preferable
  to their finite difference approximations.
*/
template<typename _T, bool _IsPeriodic = false>
class QuinticInterpolation2D :
  public QuinticInterpolation2DConstants<_T>
{

  //
  // Using from the base class.
  //
private:
  typedef QuinticInterpolation2DConstants<_T> Base;
public:
  using Base::Dimension;
protected:
  using Base::_inverseDelta;
  using Base::_offsets;

  //
  // Types.
  //
public:

  //! The value (number) type.
  typedef _T Value;
  //! The function value, gradient, and Hessian.
  typedef numerical::ValueGradientHessian<Value> ValueGradientHessian;
  //! A Cartesian point.
  typedef std::array<_T, Base::Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<_T, Base::Dimension> BBox;
  //! The %array of function values.
  typedef container::MultiArray<Value, Base::Dimension> ValueGrid;

protected:

  //! The (multi) size type.
  typedef typename ValueGrid::SizeList SizeList;
  //! The (multi) index type.
  typedef typename ValueGrid::IndexList IndexList;
  //! The single index type.
  typedef typename ValueGrid::Index Index;
  //! The 1-D derivative tensor that has up to the second derivative.
  typedef std::array<Value, 3> Dt1d3;
  //! The 1-D derivative tensor that has the function and first derivative.
  typedef std::array<Value, 2> Dt1d2;
  //! The 1-D derivative tensor that has only the function value.
  typedef Value Dt1d1;
  //! The 2-D derivative tensor has up to second derivatives.
  typedef container::EquilateralArray<Value, Base::Dimension, 3> Dt2d3;
  //! The %array of derivatives.
  typedef container::MultiArray<Dt2d3, Base::Dimension> DtGrid;

  //
  // Data.
  //
protected:

  //! The %array of 2-D derivative tensors.
  DtGrid _grid;
  //! The number of cells in each dimension.
  SizeList _cellExtents;
  //! The inverse of the number of cells in each dimension.
  /*! This quantity is stored to avoid division. */
  Point _inverseCellExtents;
  //! The lower corner of the Cartesian domain spanned by the grid.
  Point _lowerCorner;
  //! The inverse of the cell widths.
  Point _inverseWidths;

  //
  // Scratch data.
  //

  //! The interpolation cell.
  mutable IndexList _cell;
  //! Polynomial coefficients for up to quintic.
  mutable std::array<Value, 6> _c;
  //! Used for interpolation in the x dimension.
  mutable std::array<Dt1d3, 3> _y0, _y1;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and
    destructor. */
  //! @{
public:

  //! Default constructor. Invalid state.
  /*! \note You must initialize this functor before using it. */
  QuinticInterpolation2D()
  {
    // Use the default constructors for all member variables.
  }

  //! Construct from the %array of function values and the Cartesian domain.
  /*!
    \param grid The %array of function values.
    \param domain The Cartesian domain.
  */
  QuinticInterpolation2D(const ValueGrid& valueGrid, const BBox& domain)
  // Use the default constructors for all member variables.
  {
    initialize(valueGrid, domain);
  }

  //! Set the grid and domain.
  /*!
    \param grid The %array of function values.
    \param domain The Cartesian domain.
  */
  void
  initialize(const ValueGrid& valueGrid, const BBox& domain)
  {
    initialize(valueGrid, domain, std::integral_constant<bool, _IsPeriodic>());
  }

protected:

  //! Set the grid and domain for non-periodic grids.
  void
  initialize(const ValueGrid& valueGrid, const BBox& domain,
             std::false_type /*IsPeriodic*/);

#if 0
  // CONTINUE
  //! Initialize the gradient and the Hessian using the function values.
  void
  initializeNonPeriodic(const ValueGrid& g);
#endif

  //! Set the grid and domain for periodic grids.
  void
  initialize(const ValueGrid& valueGrid, const BBox& domain,
             std::true_type /*IsPeriodic*/);

  //! Initialize the function and derivatives using the extension of the input grid.
  void
  initialize(const ValueGrid& g);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  //! Interpolate the field, gradient, and Hessian at the specified point.
  void
  operator()(const Point& x, ValueGradientHessian* vgh) const
  {
    interpolate(x, vgh, std::integral_constant<std::size_t, 5>());
  }


  //! Interpolate with the template-specified order.
  template<std::size_t _Order>
  Value
  interpolate(const Point& x) const
  {
    ValueGradientHessian vgh;
    interpolate(x, &vgh, std::integral_constant<std::size_t, _Order>());
    return vgh.f;
  }

  //! Interpolate with the template-specified order. Calculate the gradient.
  template<std::size_t _Order>
  Value
  interpolate(const Point& x,
              std::array<Value, Base::Dimension>* gradient) const
  {
    ValueGradientHessian vgh;
    interpolate(x, &vgh, std::integral_constant<std::size_t, _Order>());
    (*gradient)[0] = vgh.fx;
    (*gradient)[1] = vgh.fy;
    return vgh.f;
  }

  //! Interpolate with the specified order.
  Value
  interpolate(const std::size_t order, const Point& x) const
  {
    if (order == 5) {
      return interpolate<5>(x);
    }
    if (order == 3) {
      return interpolate<3>(x);
    }
    if (order == 1) {
      return interpolate<1>(x);
    }
    if (order == 0) {
      return interpolate<0>(x);
    }
    // Other orders are not supported.
    assert(false);
    return 0;
  }

  //! Interpolate with the specified order. Calculate the gradient.
  Value
  interpolate(const std::size_t order, const Point& x,
              std::array<Value, Base::Dimension>* gradient) const
  {
    if (order == 5) {
      return interpolate<5>(x, gradient);
    }
    if (order == 3) {
      return interpolate<3>(x, gradient);
    }
    // Other orders are not supported.
    assert(false);
    return 0;
  }

  //! @}
protected:

  //! Quintic interpolation.
  /*! Pass the point by value as it will be modified. */
  void
  interpolate(Point x, ValueGradientHessian* vgh,
              std::integral_constant<std::size_t, 5> /*order*/) const;

  //! Cubic interpolation.
  /*! Pass the point by value as it will be modified. */
  void
  interpolate(Point x, ValueGradientHessian* vgh,
              std::integral_constant<std::size_t, 3> /*order*/) const;

  //! Linear interpolation.
  /*! Pass the point by value as it will be modified. */
  void
  interpolate(Point x, ValueGradientHessian* vgh,
              std::integral_constant<std::size_t, 1> /*order*/) const;

  //! Constant interpolation.
  void
  interpolate(const Point& x, ValueGradientHessian* vgh,
              std::integral_constant<std::size_t, 0> /*order*/) const;

  //! Quintic interpolation in the X direction.
  void
  interpolateX(const Dt2d3& a, const Dt2d3& b, Value t,
               std::array<Dt1d3, 3>* f,
               std::integral_constant<std::size_t, 5> /*order*/) const;

  //! Quintic interpolation in the Y direction.
  void
  interpolateY(const std::array<Dt1d3, 3>& a,
               const std::array<Dt1d3, 3>& b,
               Value t, ValueGradientHessian* f,
               std::integral_constant<std::size_t, 5> /*order*/) const;

  //! Cubic interpolation in the X direction.
  void
  interpolateX(const Dt2d3& a, const Dt2d3& b, const Value t,
               std::array<Dt1d3, 3>* f,
               std::integral_constant<std::size_t, 3> /*order*/)
  const;

  //! Cubic interpolation in the Y direction.
  void
  interpolateY(const std::array<Dt1d3, 3>& a,
               const std::array<Dt1d3, 3>& b,
               const Value t, ValueGradientHessian* f,
               std::integral_constant<std::size_t, 3> /*order*/) const;

  //! Linear interpolation in the X direction.
  void
  interpolateX(const Dt2d3& a, const Dt2d3& b, const Value t,
               std::array<Dt1d3, 3>* f,
               std::integral_constant<std::size_t, 1> /*order*/)
  const;

  //! Linear interpolation in the Y direction.
  void
  interpolateY(const std::array<Dt1d3, 3>& a,
               const std::array<Dt1d3, 3>& b,
               const Value t, ValueGradientHessian* f,
               std::integral_constant<std::size_t, 1> /*order*/) const;

  //! Calculate the coefficients for the quintic interpolant.
  void
  quinticCoefficients(Value f0, Value fx0, Value fxx0,
                      Value f1, Value fx1, Value fxx1) const;

  // Evaluate the quintic polynomial and its first and second derivatives.
  void
  evaluateQuintic(Value x, Dt1d3* f) const;

  // Evaluate the quintic polynomial and its first derivative.
  void
  evaluateQuintic(Value x, Dt1d2* f) const;

  // Evaluate the quintic polynomial.
  void
  evaluateQuintic(Value x, Dt1d1* f) const;

  //! Calculate the coefficients for the cubic interpolant.
  void
  cubicCoefficients(Value f0, Value fx0, Value f1, Value fx1) const;

  // Evaluate the cubic polynomial and its derivative.
  void
  evaluateCubic(Value x, Dt1d2* f) const;

  // Evaluate the cubic polynomial.
  void
  evaluateCubic(Value x, Dt1d1* f) const;

  //! Calculate the coefficients for the linear interpolant.
  void
  linearCoefficients(Value f0, Value f1) const;

  // Evaluate the linear polynomial.
  void
  evaluateLinear(Value x, Dt1d1* f) const;

  //! Calculate coordinates for first-order or higher interpolation.
  /*! Set the _cell member variable. \c x is used for scratch data. */
  void
  calculateScaledCoordinates(Point* x) const;

  //! Calculate the closest grid point for zeroth-order interpolation.
  /*! Set the _cell member variable. \c x is used for scratch calculations. */
  void
  snapToGrid(Point x) const;
};

} // namespace numerical
}

#define __numerical_interpolation_QuinticInterpolation2D_ipp__
#include "stlib/numerical/interpolation/QuinticInterpolation2D.ipp"
#undef __numerical_interpolation_QuinticInterpolation2D_ipp__

#endif

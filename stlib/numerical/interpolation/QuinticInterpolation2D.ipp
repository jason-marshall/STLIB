// -*- C++ -*-

#if !defined(__numerical_interpolation_QuinticInterpolation2D_ipp__)
#error This file is an implementation detail of QuinticInterpolation2D.
#endif

namespace stlib
{
namespace numerical
{

template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
initialize(const ValueGrid& valueGrid, const BBox& domain,
           std::false_type /*IsPeriodic*/)
{
  typedef container::MultiIndexRange<Base::Dimension> Range;
  typedef container::MultiIndexRangeIterator<Base::Dimension> Iterator;

  // There must be at least two grid points in each dimension.
  assert(ext::min(valueGrid.extents()) >= std::size_t(2));

  // Allocate memory for the grid.
  _grid.rebuild(valueGrid.extents());

  // We start with the array extents. In each dimension the number of cells
  // is one less than the number of elements.
  _cellExtents = _grid.extents();
  _cellExtents -= std::size_t(1);
  _lowerCorner = domain.lower;
  // Calculate the inverse cell extents and widths.
  for (std::size_t i = 0; i != Dimension; ++i) {
    _inverseCellExtents[i] = 1. / _cellExtents[i];
    _inverseWidths[i] = _cellExtents[i] / (domain.upper[i] -
                                           domain.lower[i]);
  }

  // The extension of the value grid with one layer of ghost points.
  ValueGrid extension(valueGrid.extents() + std::size_t(2),
                      IndexList{{-1, -1}});
  // Copy the values in the input grid.
  const Range range = _grid.range();
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    extension(*i) = valueGrid(*i);
  }
  // The four sides.
  {
    const Index n = valueGrid.extents()[1] - 1;
    for (Index i = 0; i != Index(valueGrid.extents()[0]); ++i) {
      extension(i, -1) = 2 * valueGrid(i, 0) - valueGrid(i, 1);
      extension(i, n + 1) = 2 * valueGrid(i, n) - valueGrid(i, n - 1);
    }
  }
  {
    const Index n = valueGrid.extents()[0] - 1;
    for (Index i = 0; i != Index(valueGrid.extents()[1]); ++i) {
      extension(-1, i) = 2 * valueGrid(0, i) - valueGrid(1, i);
      extension(n + 1, i) = 2 * valueGrid(n, i) - valueGrid(n - 1, i);
    }
  }
  // The four corners.
  {
    const Index m = valueGrid.extents()[0] - 1;
    const Index n = valueGrid.extents()[1] - 1;
    extension(-1, -1) = 3 * valueGrid(0, 0) - valueGrid(1, 0) -
                        valueGrid(0, 1);
    extension(-1, n + 1) = 3 * valueGrid(0, n) - valueGrid(1, n) -
                           valueGrid(0, n - 1);
    extension(m + 1, -1) = 3 * valueGrid(m, 0) - valueGrid(m - 1, 0) -
                           valueGrid(m, 1);
    extension(m + 1, n + 1) = 3 * valueGrid(m, n) - valueGrid(m - 1, n) -
                              valueGrid(m, n - 1);
  }

  initialize(extension);
}


// CONTINUE REMOVE
#if 0
template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
initializeNonPeriodic(const ValueGrid& g)
{
  typedef container::MultiIndexRange<Base::Dimension> Range;
  typedef container::MultiIndexRangeIterator<Base::Dimension> Iterator;

  // Closed bounds on the indices.
  const IndexList lo = _grid.bases();
  const IndexList hi = _grid.bases() +
                       ext::convert_array<Index>(_grid.extents()) - 1;

  // For each grid point.
  const Range range = _grid.range();
  const Iterator end = Iterator::end(range);
  for (Iterator iter = Iterator::begin(range); iter != end; ++iter) {
    const Index i = (*iter)[0];
    const Index j = (*iter)[1];
    ValueGradientHessian& x = _grid(i, j);
    // Value.
    x.f = g(i, j);

    // 0 indicates centered differencing.
    // 1 indicates forward differencing.
    // -1 indicates backward differencing.
    Index di = 0, dj = 0;
    if (i == _grid.bases()[0]) {
      di = 1;
    }
    else if (i == _grid.bases()[0] + Index(_grid.extents()[0]) - 1) {
      di = -1;
    }
    if (j == _grid.bases()[1]) {
      dj = 1;
    }
    else if (j == _grid.bases()[1] + Index(_grid.extents()[1]) - 1) {
      dj = -1;
    }

    // First and second derivative in x.
    if (di == 0) {
      // Use centered differencing.
      // Error = O(dx^2).
      x.fx = 0.5 * (g(i + 1, j) - g(i - 1, j));
      x.fxx = g(i + 1, j) - 2 * g(i, j) + g(i - 1, j);
    }
    else {
      // Use forward or backward differencing.
      // Check the special case that there is only one cell.
      if (_cellExtents[0] == 1) {
        // Error = O(dx).
        x.fx = di * (g(i + di, j) - g(i, j));
        // Error = O(1).
        x.fxx = 0;
      }
      else {
        // The general case uses three grid points.
        // Error = O(dx^2).
        x.fx = di * (- 0.5 * g(i + 2 * di, j) + 2 * g(i + di, j) - 1.5 * g(i, j));
        // Error = O(dx).
        x.fxx = g(i + 2 * di, j) - 2 * g(i + di, j) + g(i, j);
      }
#if 0
      // Check the special cases that there are only one or two cells.
      if (_cellExtents[0] == 1) {
        x.fx = di * (g(i + di, j) - g(i, j));
        x.fxx = 0;
      }
      else if (_cellExtents[0] == 2) {
        x.fx = di * (- 0.5 * g(i + 2 * di, j) + 2 * g(i + di, j) - 1.5 * g(i, j));
        x.fxx = g(i + 2 * di, j) - 2 * g(i + di, j) + g(i, j);
      }
      else {
        // The general case uses four grid points.
        x.fx = di * (- 0.5 * g(i + 2 * di, j) + 2 * g(i + di, j) - 1.5 * g(i, j));
        x.fxx = - g(i + 3 * di, j) + 4 * g(i + 2 * di, j) - 5 * g(i + di, j) +
                2 * g(i, j);
      }
#endif
    }

    // First and second derivative in y.
    if (dj == 0) {
      // Use centered differencing.
      // Error = O(dx^2).
      x.fy = 0.5 * (g(i, j + 1) - g(i, j - 1));
      x.fyy = g(i, j + 1) - 2 * g(i, j) + g(i, j - 1);
    }
    else {
      // Use forward or backward differencing.
      // Check the special case that there is only one cell.
      if (_cellExtents[0] == 1) {
        // Error = O(dx).
        x.fy = dj * (g(i, j + dj) - g(i, j));
        // Error = O(1).
        x.fyy = 0;
      }
      else {
        // The general case uses three grid points.
        // Error = O(dx^2).
        x.fy = dj * (- 0.5 * g(i, j + 2 * dj) + 2 * g(i, j + dj) - 1.5 * g(i, j));
        // Error = O(dx).
        x.fyy = g(i, j + 2 * dj) - 2 * g(i, j + dj) + g(i, j);
      }
#if 0
      // Check the special cases that there are only one or two cells.
      if (_cellExtents[0] == 1) {
        x.fy = dj * (g(i, j + dj) - g(i, j));
        x.fyy = 0;
      }
      else if (_cellExtents[0] == 2) {
        x.fy = dj * (- 0.5 * g(i, j + 2 * dj) + 2 * g(i, j + dj) - 1.5 * g(i, j));
        x.fyy = g(i, j + 2 * dj) - 2 * g(i, j + dj) + g(i, j);
      }
      else {
        // The general case uses four grid points.
        x.fy = dj * (- 0.5 * g(i, j + 2 * dj) + 2 * g(i, j + dj) - 1.5 * g(i, j));
        x.fyy = - g(i, j + 3 * dj) + 4 * g(i, j + 2 * dj) - 5 * g(i, j + dj) +
                2 * g(i, j);
      }
#endif
    }

    // Cross derivative.
    if (di == 0) {
      if (dj == 0) {
        // Error = O(dx^2).
        x.fxy = 0.25 * (g(i + 1, j + 1) - g(i + 1, j - 1) -
                        g(i - 1, j + 1) + g(i - 1, j - 1));
      }
      else { // dj != 0
        // Error = O(dx).
        x.fxy = 0.5 * dj * (g(i + 1, j + dj) - g(i + 1, j) -
                            g(i - 1, j + dj) + g(i - 1, j));
      }
    }
    else { // di != 0
      if (dj == 0) {
        // Error = O(dx).
        x.fxy = 0.5 * di * (g(i + di, j + 1) - g(i + di, j - 1) -
                            g(i, j + 1) + g(i, j - 1));
#if 0
        x.fxy = 0.5 * di * ((- 0.5 * g(i + 2 * di, j + 1) + 2 * g(i + di, j + 1) -
                             1.5 * g(i, j + 1)) -
                            (- 0.5 * g(i + 2 * di, j - 1) + 2 * g(i + di, j - 1) -
                             1.5 * g(i, j - 1)));
#endif
      }
      else { // dj != 0
        // Error = O(dx).
        x.fxy = di * dj * (g(i + di, j + dj) - g(i + di, j) -
                           g(i, j + dj) + g(i, j));
      }
    }
  }
}
#endif

template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
initialize(const ValueGrid& valueGrid, const BBox& domain,
           std::true_type /*IsPeriodic*/)
{
  typedef container::MultiIndexRangeIterator<Base::Dimension> Iterator;

  // There must be at least one grid point in each dimension.
  assert(ext::min(valueGrid.extents()) >= std::size_t(1));

  // Allocate memory for the grid.
  _grid.rebuild(valueGrid.extents() + std::size_t(1));
  // In each dimension the number of cells is equal to the number of elements
  // in the input grid.
  _cellExtents = valueGrid.extents();
  _lowerCorner = domain.lower;
  // Calculate the inverse cell extents and widths.
  for (std::size_t i = 0; i != Dimension; ++i) {
    _inverseCellExtents[i] = 1. / _cellExtents[i];
    _inverseWidths[i] = _cellExtents[i] / (domain.upper[i] -
                                           domain.lower[i]);
  }

  // The periodic extension of the value grid with one layer of ghost points.
  ValueGrid extension(valueGrid.extents() + std::size_t(3),
                      IndexList{{-1, -1}});
  const SizeList& e = valueGrid.extents();
  const container::MultiIndexRange<Base::Dimension> range = extension.range();
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    extension(*i) = valueGrid(((*i)[0] + e[0]) % e[0],
                              ((*i)[1] + e[1]) % e[1]);
  }

  initialize(extension);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
initialize(const ValueGrid& g)
{
  typedef container::MultiIndexRange<Base::Dimension> Range;
  typedef container::MultiIndexRangeIterator<Base::Dimension> Iterator;

  // The periodic extension of the input grid has one layer of ghost points.
  assert(_grid.extents() + std::size_t(2) == g.extents());

  // For each grid point.
  const Range range = _grid.range();
  const Iterator end = Iterator::end(range);
  for (Iterator iter = Iterator::begin(range); iter != end; ++iter) {
    const Index i = (*iter)[0];
    const Index j = (*iter)[1];
    Dt2d3& f = _grid(i, j);
    // Use second order accurate, centered difference formulas for all
    // derivatives.
    // f
    f(0, 0) = g(i, j);
    // f_x
    f(1, 0) = 0.5 * (g(i + 1, j) - g(i - 1, j));
    // f_xx
    f(2, 0) = g(i + 1, j) - 2 * g(i, j) + g(i - 1, j);
    // f_y
    f(0, 1) = 0.5 * (g(i, j + 1) - g(i, j - 1));
    // f_xy
    f(1, 1) = 0.25 * (g(i + 1, j + 1) - g(i + 1, j - 1) -
                      g(i - 1, j + 1) + g(i - 1, j - 1));
    // f_xxy
    f(2, 1) = 0.5 * ((g(i + 1, j + 1) - 2 * g(i, j + 1) + g(i - 1, j + 1)) -
                     (g(i + 1, j - 1) - 2 * g(i, j - 1) + g(i - 1, j - 1)));
    // f_yy
    f(0, 2) = g(i, j + 1) - 2 * g(i, j) + g(i, j - 1);
    // f_xyy
    f(1, 2) = 0.5 * ((g(i + 1, j + 1) - 2 * g(i + 1, j) + g(i + 1, j - 1)) -
                     (g(i - 1, j + 1) - 2 * g(i - 1, j) + g(i - 1, j - 1)));
    // f_xxyy
    f(2, 2) = g(i + 1, j + 1) - 2 * g(i + 1, j) + g(i + 1, j - 1) -
              2 * (g(i, j + 1) - 2 * g(i, j) + g(i, j - 1)) +
              g(i - 1, j + 1) - 2 * g(i - 1, j) + g(i - 1, j - 1);
  }
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolate(Point x, ValueGradientHessian* vgh,
            std::integral_constant<std::size_t, 5> /*order*/) const
{
  // Find the appropriate cell.
  calculateScaledCoordinates(&x);

  Dt2d3 a, b;
  // Interpolate in the x direction along the line y = 0.
  {
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y0, std::integral_constant<std::size_t, 5>());
  }

  // Interpolate in the x direction along the line y = 1.
  {
    _cell[0] -= 1;
    _cell[1] += 1;
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y1, std::integral_constant<std::size_t, 5>());
  }

  // Interpolate in the y direction.
  interpolateY(_y0, _y1, x[1], vgh, std::integral_constant<std::size_t, 5>());

  // Scale to go from logical coordinates to physical coordinates.
  vgh->fx *= _inverseWidths[0];
  vgh->fy *= _inverseWidths[1];
  vgh->fxx *= _inverseWidths[0] * _inverseWidths[0];
  vgh->fxy *= _inverseWidths[0] * _inverseWidths[1];
  vgh->fyy *= _inverseWidths[1] * _inverseWidths[1];
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolate(Point x, ValueGradientHessian* vgh,
            std::integral_constant<std::size_t, 3> /*order*/) const
{
  // Find the appropriate cell.
  calculateScaledCoordinates(&x);

  // Interpolate in the x direction along the line y = 0.
  {
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y0, std::integral_constant<std::size_t, 3>());
  }

  // Interpolate in the x direction along the line y = 1.
  {
    _cell[0] -= 1;
    _cell[1] += 1;
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y1, std::integral_constant<std::size_t, 3>());
  }

  // Interpolate in the y direction.
  interpolateY(_y0, _y1, x[1], vgh, std::integral_constant<std::size_t, 3>());

  // Scale to go from logical coordinates to physical coordinates.
  vgh->fx *= _inverseWidths[0];
  vgh->fy *= _inverseWidths[1];
  // Ensure that the second derivatives are not used.
  vgh->fxx = std::numeric_limits<Value>::quiet_NaN();
  vgh->fxy = std::numeric_limits<Value>::quiet_NaN();
  vgh->fyy = std::numeric_limits<Value>::quiet_NaN();
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolate(Point x, ValueGradientHessian* vgh,
            std::integral_constant<std::size_t, 1> /*order*/) const
{
  // Find the appropriate cell.
  calculateScaledCoordinates(&x);

  // Interpolate in the x direction along the line y = 0.
  {
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y0, std::integral_constant<std::size_t, 1>());
  }

  // Interpolate in the x direction along the line y = 1.
  {
    _cell[0] -= 1;
    _cell[1] += 1;
    const Dt2d3& a = _grid(_cell);
    _cell[0] += 1;
    const Dt2d3& b = _grid(_cell);
    interpolateX(a, b, x[0], &_y1, std::integral_constant<std::size_t, 1>());
  }

  // Interpolate in the y direction.
  interpolateY(_y0, _y1, x[1], vgh, std::integral_constant<std::size_t, 1>());

  // Ensure that the derivatives are not used.
  vgh->fx = std::numeric_limits<Value>::quiet_NaN();
  vgh->fy = std::numeric_limits<Value>::quiet_NaN();
  vgh->fxx = std::numeric_limits<Value>::quiet_NaN();
  vgh->fxy = std::numeric_limits<Value>::quiet_NaN();
  vgh->fyy = std::numeric_limits<Value>::quiet_NaN();
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolate(const Point& x, ValueGradientHessian* vgh,
            std::integral_constant<std::size_t, 0> /*order*/) const
{
  // Find the appropriate grid point.
  snapToGrid(x);
  // Set the function value.
  vgh->f = _grid(_cell)(0, 0);
  // Ensure that the derivatives are not used.
  vgh->fx = std::numeric_limits<Value>::quiet_NaN();
  vgh->fy = std::numeric_limits<Value>::quiet_NaN();
  vgh->fxx = std::numeric_limits<Value>::quiet_NaN();
  vgh->fxy = std::numeric_limits<Value>::quiet_NaN();
  vgh->fyy = std::numeric_limits<Value>::quiet_NaN();
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateX(const Dt2d3& a, const Dt2d3& b, const Value t,
             std::array<Dt1d3, 3>* f,
             std::integral_constant<std::size_t, 5> /*order*/) const
{
  // For f, f_y, and f_yy.
  for (std::size_t j = 0; j != 3; ++j) {
    quinticCoefficients(a(0, j), a(1, j), a(2, j), b(0, j), b(1, j), b(2, j));
    for (std::size_t i = 0; i != _offsets.size(); ++i) {
      evaluateQuintic(t + _offsets[i], &(*f)[i][j]);
    }
  }
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateY(const std::array<Dt1d3, 3>& a,
             const std::array<Dt1d3, 3>& b,
             const Value t, ValueGradientHessian* f,
             std::integral_constant<std::size_t, 5> /*order*/) const
{
  // We need the following information to compute the Hessian.
  // 0         1       2
  // f(x-D)    f(x)    f(x+D)
  // f_y(x-D)  f_y(x)  f_y(x+D)
  //           f_yy(x)
  Dt1d2 y0, y2;
  Dt1d3 y1;
  quinticCoefficients(a[0][0], a[0][1], a[0][2], b[0][0], b[0][1], b[0][2]);
  evaluateQuintic(t, &y0);
  quinticCoefficients(a[1][0], a[1][1], a[1][2], b[1][0], b[1][1], b[1][2]);
  evaluateQuintic(t, &y1);
  quinticCoefficients(a[2][0], a[2][1], a[2][2], b[2][0], b[2][1], b[2][2]);
  evaluateQuintic(t, &y2);

  f->f = y1[0];
  f->fx = 0.5 * (y2[0] - y0[0]) * _inverseDelta;
  f->fy = y1[1];
  f->fxx = (y2[0] - 2 * y1[0] + y0[0]) * _inverseDelta * _inverseDelta;
  f->fxy = 0.5 * (y2[1] - y0[1]) * _inverseDelta;
  f->fyy = y1[2];
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateX(const Dt2d3& a, const Dt2d3& b, const Value t,
             std::array<Dt1d3, 3>* f,
             std::integral_constant<std::size_t, 3> /*order*/) const
{
  // For f and f_y.
  for (std::size_t j = 0; j != 2; ++j) {
    cubicCoefficients(a(0, j), a(1, j), b(0, j), b(1, j));
    // For -d, 0, and d.
    for (std::size_t i = 0; i != _offsets.size(); ++i) {
      evaluateCubic(t + _offsets[i], &(*f)[i][j]);
    }
  }
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateY(const std::array<Dt1d3, 3>& a,
             const std::array<Dt1d3, 3>& b,
             const Value t, ValueGradientHessian* f,
             std::integral_constant<std::size_t, 3> /*order*/) const
{
  // We need the following information to compute the function and first
  // derivatives.
  // 0         1       2
  // f(x-D)    f(x)    f(x+D)
  //           f_y(x)
  Dt1d1 y0, y2;
  Dt1d2 y1;
  cubicCoefficients(a[0][0], a[0][1], b[0][0], b[0][1]);
  evaluateCubic(t, &y0);
  cubicCoefficients(a[1][0], a[1][1], b[1][0], b[1][1]);
  evaluateCubic(t, &y1);
  cubicCoefficients(a[2][0], a[2][1], b[2][0], b[2][1]);
  evaluateCubic(t, &y2);

  f->f = y1[0];
  f->fx = 0.5 * (y2 - y0) * _inverseDelta;
  f->fy = y1[1];
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateX(const Dt2d3& a, const Dt2d3& b, const Value t,
             std::array<Dt1d3, 3>* f,
             std::integral_constant<std::size_t, 1> /*order*/) const
{
  linearCoefficients(a(0, 0), b(0, 0));
  // Use only the middle tensor.
  evaluateLinear(t, &(*f)[1][0]);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
interpolateY(const std::array<Dt1d3, 3>& a,
             const std::array<Dt1d3, 3>& b,
             const Value t, ValueGradientHessian* f,
             std::integral_constant<std::size_t, 1> /*order*/) const
{
  linearCoefficients(a[1][0], b[1][0]);
  evaluateLinear(t, &f->f);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
quinticCoefficients(const Value f0, const Value fx0, const Value fxx0,
                    const Value f1, const Value fx1, const Value fxx1) const
{
  _c[0] = f0;
  _c[1] = fx0;
  _c[2] = 0.5 * fxx0;
  _c[3] = 0.5 * (-20 * f0 + 20 * f1 - 12 * fx0 - 8 * fx1 - 3 * fxx0 + fxx1);
  _c[4] = 15 * f0 - 15 * f1 + 8 * fx0 + 7 * fx1 + 1.5 * fxx0 - fxx1;
  _c[5] = 0.5 * (-12 * f0 + 12 * f1 - 6 * fx0 - 6 * fx1 - fxx0 + fxx1);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateQuintic(const Value x, Dt1d3* f) const
{
  // Evaluate f().
  (*f)[0] = evaluatePolynomial<5>(_c.begin(), x);
  // Differentiate.
  differentiatePolynomialCoefficients<5>(_c.begin());
  // Evaluate f'().
  (*f)[1] = evaluatePolynomial<4>(_c.begin(), x);
  // Differentiate.
  differentiatePolynomialCoefficients<4>(_c.begin());
  // Evaluate f''().
  (*f)[2] = evaluatePolynomial<3>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateQuintic(const Value x, Dt1d2* f) const
{
  // Evaluate f().
  (*f)[0] = evaluatePolynomial<5>(_c.begin(), x);
  // Differentiate.
  differentiatePolynomialCoefficients<5>(_c.begin());
  // Evaluate f'().
  (*f)[1] = evaluatePolynomial<4>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateQuintic(const Value x, Dt1d1* f) const
{
  // Evaluate f().
  *f = evaluatePolynomial<5>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
cubicCoefficients(const Value f0, const Value fx0, const Value f1,
                  const Value fx1) const
{
  _c[0] = f0;
  _c[1] = fx0;
  _c[2] = 3 * (f1 - f0) - 2 * fx0 - fx1;
  _c[3] = 2 * (f0 - f1) + fx0 + fx1;
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateCubic(const Value x, Dt1d2* f) const
{
  // Evaluate f().
  (*f)[0] = evaluatePolynomial<3>(_c.begin(), x);
  // Differentiate.
  differentiatePolynomialCoefficients<3>(_c.begin());
  // Evaluate f'().
  (*f)[1] = evaluatePolynomial<2>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateCubic(const Value x, Dt1d1* f) const
{
  // Evaluate f().
  *f = evaluatePolynomial<3>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
linearCoefficients(const Value f0, const Value f1) const
{
  _c[0] = f0;
  _c[1] = f1 - f0;
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
evaluateLinear(const Value x, Dt1d1* f) const
{
  // Evaluate f().
  *f = evaluatePolynomial<1>(_c.begin(), x);
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
calculateScaledCoordinates(Point* x) const
{
  // Convert to index coordinates.
  *x -= _lowerCorner;
  *x *= _inverseWidths;
  if (_IsPeriodic) {
    // Use periodicity to transform to the canonical domain.
    for (std::size_t i = 0; i != Dimension; ++i) {
      (*x)[i] -= std::floor((*x)[i] * _inverseCellExtents[i]) *
                 _cellExtents[i];
    }
  }
  // Use casting to determine the cell.
  for (std::size_t i = 0; i != Dimension; ++i) {
    _cell[i] = static_cast<Index>((*x)[i]);
  }
  assert(0 <= _cell[0] && _cell[0] <= Index(_cellExtents[0]) &&
         0 <= _cell[1] && _cell[1] <= Index(_cellExtents[1]));
  // Convert to the cell parametrization, which is [0..1)^_Dimension.
  for (std::size_t i = 0; i != Dimension; ++i) {
    (*x)[i] -= _cell[i];
  }
#ifdef STLIB_DEBUG
  assert(0 <= (*x)[0] && (*x)[0] < 1 && 0 <= (*x)[1] && (*x)[1] < 1);
#endif
}


template<typename _T, bool _IsPeriodic>
inline
void
QuinticInterpolation2D<_T, _IsPeriodic>::
snapToGrid(Point x) const
{
  // Convert to index coordinates.
  x -= _lowerCorner;
  x *= _inverseWidths;
  // Add 0.5 to move from grid cells to grid points.
  x += 0.5;
  if (_IsPeriodic) {
    // Use periodicity to transform to the canonical domain.
    for (std::size_t i = 0; i != Dimension; ++i) {
      x[i] -= std::floor(x[i] * _inverseCellExtents[i]) * _cellExtents[i];
    }
  }
  // Use casting to determine the grid point.
  for (std::size_t i = 0; i != Dimension; ++i) {
    _cell[i] = static_cast<Index>(x[i]);
  }
  assert(0 <= _cell[0] && _cell[0] < Index(_grid.extents()[0]) &&
         0 <= _cell[1] && _cell[1] < Index(_grid.extents()[1]));
}

} // namespace numerical
}

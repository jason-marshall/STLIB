// -*- C++ -*-

#if !defined(__numerical_interpolation_InterpolatingFunctionRegularGridBase_ipp__)
#error This file is an implementation detail of InterpolatingFunctionRegularGridBase.
#endif

namespace stlib
{
namespace numerical
{

namespace internal
{


//-----------------------------------------------------------------------------
// Zeroth-order interpolation in N-D.
//-----------------------------------------------------------------------------


template<typename _T, std::size_t _Dimension>
inline
_T
interpolateFunction(container::EquilateralArray<_T, _Dimension, 1> f,
                    const std::array<double, _Dimension>& /*x*/)
{
  return f[0];
}


//-----------------------------------------------------------------------------
// Multi-linear interpolation in N-D.
//-----------------------------------------------------------------------------


// N-D, linear interpolation.
// Note the the first argument is passed by value as it is modified.
// Let i be a multi-index and n be the dimension index.
// The linear interpolation formula is
// sum_i prod_n c_{n i_n} f_i
// where c_nj = delta_j0 (1 - x_n) + delta_j1 x_n.
template<typename _T, std::size_t _Dimension>
inline
_T
interpolateFunction(container::EquilateralArray<_T, _Dimension, 2> f,
                    const std::array<double, _Dimension>& x)
{
  const std::size_t Size = std::size_t(1) << _Dimension;
  for (std::size_t d = 0; d != _Dimension; ++d) {
    const double values[] = {(1 - x[d]), x[d]};
    const std::size_t mask = 1 << d;
    for (std::size_t i = 0; i != Size; ++i) {
      f[i] *= values[(i & mask) >> d];
    }
  }
  return ext::sum(f);
}


//-----------------------------------------------------------------------------
// Cubic interpolation in 1-D.
//-----------------------------------------------------------------------------


// 1-D, cubic interpolation.
template<typename _T>
inline
_T
interpolateFunction(const std::array<_T, 2>& f,
                    const std::array<_T, 2>& d,
                    const std::array<double, 1>& x)
{
  const _T t = x[0];
  const _T t2 = t * t;
  const _T t3 = t2 * t;
  return f[0] * (2 * t3 - 3 * t2 + 1) +
         f[1] * (-2 * t3 + 3 * t2) +
         d[0] * (t3 - 2 * t2 + t) +
         d[1] * (t3 - t2);
}

// 1-D, cubic interpolation of function and derivative.
template<typename _T>
inline
_T
interpolateFunction(const std::array<_T, 2>& f,
                    const std::array<_T, 2>& d,
                    const std::array<double, 1>& x,
                    std::array<_T, 1>* gradient)
{
  const _T t = x[0];
  const _T t2 = t * t;
  const _T t3 = t2 * t;
  (*gradient)[0] =
    (f[0] - f[1]) * 6 * (t2 - t) +
    d[0] * (3 * t2 - 4 * t + 1) +
    d[1] * (3 * t2 - 2 * t);
  return f[0] * (2 * t3 - 3 * t2 + 1) +
         f[1] * (-2 * t3 + 3 * t2) +
         d[0] * (t3 - 2 * t2 + t) +
         d[1] * (t3 - t2);
}

// 1-D, cubic interpolation of function value.
template<typename _T>
inline
_T
interpolateFunction(const container::EquilateralArray<_T, 1, 4>& y,
                    const std::array<double, 1>& x)
{
  // Extract the function values and compute the gradients.
  std::array<_T, 2> f = {{y[1], y[2]}};
  std::array<_T, 2> d = {{0.5 * (y[2] - y[0]), 0.5 * (y[3] - y[1])}};
  // Interpolate.
  return interpolateFunction(f, d, x);
}


// 1-D, cubic interpolation of function and gradient.
template<typename _T>
inline
_T
interpolateFunction(const container::EquilateralArray<_T, 1, 4>& y,
                    const std::array<double, 1>& x,
                    std::array<_T, 1>* gradient)
{
  // Extract the function values and compute the gradients.
  std::array<_T, 2> f = {{y[1], y[2]}};
  std::array<_T, 2> d = {{0.5 * (y[2] - y[0]), 0.5 * (y[3] - y[1])}};
  // Interpolate.
  return interpolateFunction(f, d, x, gradient);
}


//-----------------------------------------------------------------------------
// Bicubic interpolation in 2-D.
//-----------------------------------------------------------------------------


template<typename _T>
inline
void
calculateDerivatives(const container::EquilateralArray<_T, 2, 4>& f,
                     container::EquilateralArray<_T, 2, 2>* y,
                     container::EquilateralArray<_T, 2, 2>* y1,
                     container::EquilateralArray<_T, 2, 2>* y2,
                     container::EquilateralArray<_T, 2, 2>* y12)
{
  // From the 2-D array of function values compute packed 1-D arrays of the
  // function values and derivatives.
  // Function, first derivatives, and cross derivative.
  for (std::size_t j = 1; j != 3; ++j) {
    for (std::size_t i = 1; i != 3; ++i) {
      (*y)(i - 1, j - 1) = f(i, j);
      (*y1)(i - 1, j - 1) = 0.5 * (f(i + 1, j) - f(i - 1, j));
      (*y2)(i - 1, j - 1) = 0.5 * (f(i, j + 1) - f(i, j - 1));
      (*y12)(i - 1, j - 1) = 0.25 * (f(i + 1, j + 1) - f(i + 1, j - 1) - f(i - 1,
                                     j + 1)
                                     + f(i - 1, j - 1));
    }
  }
}


// 2-D, bicubic interpolation coefficients.
template<typename _T>
inline
void
calculateInterpolationCoefficients(const container::EquilateralArray<_T, 2, 4>&
                                   f,
                                   container::EquilateralArray<_T, 2, 4>* c)
{
  // Static data.
  static container::EquilateralArray<_T, 2, 2> y, y1, y2, y12;
  BOOST_STATIC_CONSTEXPR double wt[16][16] =
    {{1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.},
    {-3., 0., 0., 3., 0., 0., 0., 0., -2., 0., 0., -1., 0., 0., 0., 0.},
    {2., 0., 0., -2., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.},
    {0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.},
    {0., 0., 0., 0., -3., 0., 0., 3., 0., 0., 0., 0., -2., 0., 0., -1.},
    {0., 0., 0., 0., 2., 0., 0., -2., 0., 0., 0., 0., 1., 0., 0., 1.},
    {-3., 3., 0., 0., -2., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    {0., 0., 0., 0., 0., 0., 0., 0., -3., 3., 0., 0., -2., -1., 0., 0.},
    {9., -9., 9., -9., 6., 3., -3., -6., 6., -6., -3., 3., 4., 2., 1., 2.},
    {-6., 6., -6., 6., -4., -2., 2., 4., -3., 3., 3., -3., -2., -1., -1., -2.},
    {2., -2., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
    {0., 0., 0., 0., 0., 0., 0., 0., 2., -2., 0., 0., 1., 1., 0., 0.},
    {-6., 6., -6., 6., -3., -3., 3., 3., -4., 4., 2., -2., -2., -2., -1., -1.},
    {4., -4., 4., -4., 2., 2., -2., -2., 2., -2., -2., 2., 1., 1., 1., 1.}
  };
  // Calculate derivatives.
  calculateDerivatives(f, &y, &y1, &y2, &y12);
  // Pack a temporary vector with a non-standard order to comply with
  // the NR algorithm.
  _T v[] = {y[0], y[1], y[3], y[2],
            y1[0], y1[1], y1[3], y1[2],
            y2[0], y2[1], y2[3], y2[2],
            y12[0], y12[1], y12[3], y12[2]
           };
  // Matrix multiply.
  // Initialize to zero.
  double a[16] = {};
  for (std::size_t i = 0; i != 16; ++i) {
    for (std::size_t j = 0; j != 16; ++j) {
      a[i] += wt[i][j] * v[j];
    }
  }
  // Unpack the coefficients.
  std::size_t n = 0;
  for (std::size_t i = 0; i != 4; ++i) {
    for (std::size_t j = 0; j != 4; ++j) {
      (*c)(i, j) = a[n++];
    }
  }
}


// 2-D, bicubic interpolation of function.
template<typename _T>
inline
_T
interpolateUsingCoefficients(const container::EquilateralArray<_T, 2, 4>& c,
                             const std::array<double, 2>& x)
{
  _T f = 0;
  for (std::size_t i = 4; i != 0;) {
    --i;
    f = x[0] * f + ((c(i, 3) * x[1] + c(i, 2)) * x[1] + c(i, 1)) * x[1] + c(i, 0);
  }
  return f;
}


// 2-D, bicubic interpolation of function and gradient.
template<typename _T>
inline
_T
interpolateUsingCoefficients(const container::EquilateralArray<_T, 2, 4>& c,
                             const std::array<double, 2>& x,
                             std::array<_T, 2>* gradient)
{
  _T f = 0;
  std::fill(gradient->begin(), gradient->end(), 0);
  for (std::size_t i = 4; i != 0;) {
    --i;
    f = x[0] * f + ((c(i, 3) * x[1] + c(i, 2)) * x[1] + c(i, 1)) * x[1] + c(i, 0);
    (*gradient)[0] = x[1] * (*gradient)[0]
                     + (3 * c(3, i) * x[0] + 2 * c(2, i)) * x[0] + c(1, i);
    (*gradient)[1] = x[0] * (*gradient)[1]
                     + (3 * c(i, 3) * x[1] + 2 * c(i, 2)) * x[1] + c(i, 1);
  }
  return f;
}


// 2-D, bicubic interpolation of function.
template<typename _T>
inline
_T
interpolateFunction(const container::EquilateralArray<_T, 2, 4>& f,
                    const std::array<double, 2>& x)
{
  static container::EquilateralArray<_T, 2, 4> c;
  // Calculate the interpolation coefficients.
  calculateInterpolationCoefficients(f, &c);
  // Interpolate.
  return interpolateUsingCoefficients(c, x);
}


// 2-D, bicubic interpolation of function and gradient.
template<typename _T>
inline
_T
interpolateFunction(const container::EquilateralArray<_T, 2, 4>& f,
                    const std::array<double, 2>& x,
                    std::array<_T, 2>* gradient)
{
  static container::EquilateralArray<_T, 2, 4> c;
  // Calculate the interpolation coefficients.
  calculateInterpolationCoefficients(f, &c);
  // Interpolate.
  return interpolateUsingCoefficients(c, x, gradient);
}


//-----------------------------------------------------------------------------
// Extrapolation to fill in missing values. This is used when extracting
// cells for cubic interpolation on a non-periodic grid.
//-----------------------------------------------------------------------------

// Linear extrapolation.
template<typename _T>
inline
void
extrapolateGrid(container::EquilateralArray<_T, 1, 4>* y)
{
  if ((*y)[0] == std::numeric_limits<_T>::max()) {
    (*y)[0] = 2 * (*y)[1] - (*y)[2];
  }
  if ((*y)[3] == std::numeric_limits<_T>::max()) {
    (*y)[3] = 2 * (*y)[2] - (*y)[1];
  }
}

// Bilinear extrapolation.
template<typename _T>
inline
void
extrapolateGrid(container::EquilateralArray<_T, 2, 4>* y)
{
  std::array<std::array<std::size_t, 2>, 12> indices = {{ {{0, 0}}, {{1, 0}}, {{2, 0}}, {{3, 0}},
      {{0, 1}}, {{3, 1}}, {{0, 2}}, {{3, 2}},
      {{0, 3}}, {{1, 3}}, {{2, 3}}, {{3, 3}}
    }
  };
  container::EquilateralArray<_T, 2, 2> f;
  f[0] = (*y)(1, 1);
  f[1] = (*y)(2, 1);
  f[2] = (*y)(1, 2);
  f[3] = (*y)(2, 2);
  for (std::size_t i = 0; i != indices.size(); ++i) {
    if ((*y)(indices[i]) == std::numeric_limits<_T>::max()) {
      std::array<double, 2> x = {{
          double(indices[i][0]) - 1,
          double(indices[i][1]) - 1
        }
      };
      (*y)(indices[i]) = interpolateFunction(f, x);
    }
  }
}


//----------------------------------------------------------------------------
// Extract the relevant cells. For linear interpolation we need to extract
// the cell (voxel) that contains the interpolation point. For cubic
// interpolation we need one additional cell in each direction in
// order to compute function derivatives.
//----------------------------------------------------------------------------


// Zeroth-order.
template<typename _T, std::size_t _Dimension, typename _IsPeriodic>
inline
void
extractCells(const container::MultiArrayConstRef<_T, _Dimension>& grid,
             const typename container::MultiArrayConstRef<_T, _Dimension>::
             IndexList& i,
             container::EquilateralArray<_T, _Dimension, 1>* y,
             _IsPeriodic /*dummy*/)
{
  (*y)[0] = grid(i);
}


template<typename _T, std::size_t _Dimension>
inline
void
extractCells(const container::MultiArrayConstRef<_T, _Dimension>& grid,
             const typename container::MultiArrayConstRef<_T, _Dimension>::
             IndexList& i,
             container::EquilateralArray<_T, _Dimension, 2>* y,
             std::false_type /*IsPeriodic*/)
{
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::SizeList
    SizeList;
  typedef container::MultiIndexRangeIterator<_Dimension> Iterator;

#ifdef STLIB_DEBUG
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::IndexList
    IndexList;
  assert(grid.bases() == ext::filled_array<IndexList>(0));
  for (std::size_t n = 0; n != _Dimension; ++n) {
    assert(0 <= i[n] && std::size_t(i[n] + 1) < grid.extents()[n]);
  }
#endif

  // Copy function values into a fixed-size array.
  const container::MultiIndexRange<_Dimension>
  range(ext::filled_array<SizeList>(2), i);
  std::size_t n = 0;
  const Iterator end = Iterator::end(range);
  for (Iterator index = Iterator::begin(range); index != end; ++index) {
    (*y)[n++] = grid(*index);
  }
}


template<typename _T, std::size_t _Dimension>
inline
void
extractCells(const container::MultiArrayConstRef<_T, _Dimension>& grid,
             const typename container::MultiArrayConstRef<_T, _Dimension>::
             IndexList& i,
             container::EquilateralArray<_T, _Dimension, 2>* y,
             std::true_type /*IsPeriodic*/)
{
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::SizeList
  SizeList;
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::IndexList
  IndexList;
  typedef container::MultiIndexRangeIterator<_Dimension> Iterator;

#ifdef STLIB_DEBUG
  assert(grid.bases() == ext::filled_array<IndexList>(0));
  for (std::size_t n = 0; n != _Dimension; ++n) {
    assert(0 <= i[n] && std::size_t(i[n]) < grid.extents()[n]);
  }
#endif

  // Copy function values into a fixed-size array.
  const container::MultiIndexRange<_Dimension>
  range(ext::filled_array<SizeList>(2), i);
  IndexList j;
  std::size_t n = 0;
  const Iterator end = Iterator::end(range);
  for (Iterator index = Iterator::begin(range); index != end; ++index) {
    j = *index;
    for (std::size_t k = 0; k != j.size(); ++k) {
      j[k] %= grid.extents()[k];
    }
    (*y)[n++] = grid(j);
  }
}


template<typename _T, std::size_t _Dimension>
inline
void
extractCells(const container::MultiArrayConstRef<_T, _Dimension>& grid,
             const typename container::MultiArrayConstRef<_T, _Dimension>::
             IndexList& i,
             container::EquilateralArray<_T, _Dimension, 4>* y,
             std::false_type /*IsPeriodic*/)
{
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::SizeList
  SizeList;
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::IndexList
  IndexList;
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::Index Index;
  typedef container::MultiIndexRangeIterator<_Dimension> Iterator;

#ifdef STLIB_DEBUG
  assert(grid.bases() == ext::filled_array<IndexList>(0));
  for (std::size_t n = 0; n != _Dimension; ++n) {
    assert(0 <= i[n] && std::size_t(i[n] + 1) < grid.extents()[n]);
  }
#endif

  // Copy function values into a fixed-size array.
  // First determine which points in the 4^_Dimension stencil overlap the grid.
  std::fill(y->begin(), y->end(), std::numeric_limits<_T>::max());
  SizeList extents = ext::filled_array<SizeList>(4);
  IndexList bases = ext::filled_array<IndexList>(0);
  bool needsExtrapolation = false;
  for (std::size_t n = 0; n != _Dimension; ++n) {
    if (i[n] == 0) {
      --extents[n];
      ++bases[n];
      needsExtrapolation = true;
    }
    if (std::size_t(i[n] + 2) == grid.extents()[n]) {
      --extents[n];
      needsExtrapolation = true;
    }
  }
  // Then copy the overlapping values.
  const container::MultiIndexRange<_Dimension> range(extents, bases);
  const Iterator end = Iterator::end(range);
  IndexList j;
  for (Iterator index = Iterator::begin(range); index != end; ++index) {
    j = i;
    j += *index;
    j -= Index(1);
    (*y)(*index) = grid(j);
  }
  // Extrapolate to fill in the missing values.
  if (needsExtrapolation) {
    extrapolateGrid(y);
  }
}


template<typename _T, std::size_t _Dimension>
inline
void
extractCells(const container::MultiArrayConstRef<_T, _Dimension>& grid,
             const typename container::MultiArrayConstRef<_T, _Dimension>::
             IndexList& i,
             container::EquilateralArray<_T, _Dimension, 4>* y,
             std::true_type /*IsPeriodic*/)
{
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::SizeList
  SizeList;
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::IndexList
  IndexList;
  typedef typename container::MultiArrayConstRef<_T, _Dimension>::Index Index;
  typedef container::MultiIndexRangeIterator<_Dimension> Iterator;

#ifdef STLIB_DEBUG
  assert(grid.bases() == ext::filled_array<IndexList>(0));
  for (std::size_t n = 0; n != _Dimension; ++n) {
    assert(0 <= i[n] && std::size_t(i[n]) < grid.extents()[n]);
  }
#endif

  // Copy function values into a fixed-size array.
  const container::MultiIndexRange<_Dimension>
  range(ext::filled_array<SizeList>(4), i - Index(1));
  IndexList j;
  std::size_t n = 0;
  const Iterator end = Iterator::end(range);
  for (Iterator index = Iterator::begin(range); index != end; ++index) {
    j = *index;
    // We need this step because the indices may be negative.
    for (std::size_t k = 0; k != j.size(); ++k) {
      j[k] += grid.extents()[k];
      j[k] %= grid.extents()[k];
    }
    (*y)[n++] = grid(j);
  }
}


//----------------------------------------------------------------------------
// A level of indirection for the interpolation order and periodicity.
// _Order and _IsPeriodic must specified explicitly.
//----------------------------------------------------------------------------


template<std::size_t _Order, bool _IsPeriodic, typename _T,
         std::size_t _Dimension>
inline
_T
interpolateFunction(const container::MultiArrayConstRef<_T, _Dimension>& grid,
                    const typename container::MultiArrayConstRef<_T, _Dimension>::
                    IndexList& cell,
                    const std::array<double, _Dimension>& x)
{
  // Copy function values into a fixed-size array.
  // The number of grid points in each dimension is one more than the
  // interpolation order.
  container::EquilateralArray < _T, _Dimension, _Order + 1 > y;
  extractCells(grid, cell, &y, std::integral_constant<bool, _IsPeriodic>());
  // Interpolate.
  return interpolateFunction(y, x);
}


template<std::size_t _Order, bool _IsPeriodic, typename _T,
         std::size_t _Dimension>
inline
_T
interpolateFunction(const container::MultiArrayConstRef<_T, _Dimension>& grid,
                    const typename container::MultiArrayConstRef<_T, _Dimension>::
                    IndexList& cell,
                    const std::array<double, _Dimension>& x,
                    std::array<_T, _Dimension>* gradient)
{
  // Copy function values into a fixed-size array.
  // The number of grid points in each dimension is one more than the
  // interpolation order.
  container::EquilateralArray < _T, _Dimension, _Order + 1 > y;
  extractCells(grid, cell, &y, std::integral_constant<bool, _IsPeriodic>());
  // Interpolate.
  return interpolateFunction(y, x, gradient);
}


} // namespace internal


//-----------------------------------------------------------------------------
// Member functions.
//-----------------------------------------------------------------------------


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
InterpolatingFunctionRegularGridBase
<_T, _Dimension, _DefaultOrder, _IsPeriodic>::
InterpolatingFunctionRegularGridBase() :
  _gridPointer(0),
  _cellExtents(),
  _inverseCellExtents(),
  _lowerCorner(),
  _inverseWidths()
{
  static_assert(_DefaultOrder == 0 || _DefaultOrder == 1 ||
                    ((_Dimension == 1 || _Dimension == 2) &&
                     _DefaultOrder == 3), "Not supported.");
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
InterpolatingFunctionRegularGridBase
<_T, _Dimension, _DefaultOrder, _IsPeriodic>::
InterpolatingFunctionRegularGridBase(const SizeList& extents,
                                     const BBox& domain) :
  _gridPointer(0),
  _cellExtents(extents),
  _inverseCellExtents(),
  _lowerCorner(domain.lower),
  _inverseWidths()
{
  static_assert(_DefaultOrder == 0 || _DefaultOrder == 1 ||
                    ((_Dimension == 1 || _Dimension == 2) &&
                     _DefaultOrder == 3), "Not supported.");
  // We start with the array extents. In each dimension the number of cells
  // is one less than the number of elements.
  _cellExtents -= std::size_t(1);
  // Expand by one cell in the positive direction if the grid represents
  // a periodic function.
  _cellExtents += std::size_t(_IsPeriodic);
  // Calculate the inverse cell extents and widths.
  for (std::size_t i = 0; i != _Dimension; ++i) {
    _inverseCellExtents[i] = 1. / _cellExtents[i];
    _inverseWidths[i] = _cellExtents[i] / (domain.upper[i] -
                                           domain.lower[i]);
  }
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
void
InterpolatingFunctionRegularGridBase
<_T, _Dimension, _DefaultOrder, _IsPeriodic>::
initialize(const SizeList& extents, const BBox& domain)
{
  _cellExtents = extents;
  _lowerCorner = domain.lower;
  // We start with the array extents. In each dimension the number of cells
  // is one less than the number of elements.
  _cellExtents -= std::size_t(1);
  // Expand by one cell in the positive direction if the grid represents
  // a periodic function.
  _cellExtents += std::size_t(_IsPeriodic);
  // Calculate the inverse cell extents and widths.
  for (std::size_t i = 0; i != _Dimension; ++i) {
    _inverseCellExtents[i] = 1. / _cellExtents[i];
    _inverseWidths[i] = _cellExtents[i] / (domain.upper[i] -
                                           domain.lower[i]);
  }
}

template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
template<std::size_t _Order>
inline
typename InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder,
         _IsPeriodic>::result_type
         InterpolatingFunctionRegularGridBase
         <_T, _Dimension, _DefaultOrder, _IsPeriodic>::
         interpolate(argument_type x) const
{
  // Ensure that the data for the grid array is valid.
#ifdef STLIB_DEBUG
  assert(_gridPointer);
  assert(_gridPointer->begin());
#endif
  // Calculate the scaled coordinates, in [0..1]^_Dimension, and the
  // determine the enclosing grid cell.
  IndexList cell;
  calculateScaledCoordinates<_Order>(&cell, &x);
  // Interpolate.
  return internal::interpolateFunction<_Order, _IsPeriodic>
         (*_gridPointer, cell, x);
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
template<std::size_t _Order>
inline
typename InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder,
         _IsPeriodic>::result_type
         InterpolatingFunctionRegularGridBase
         <_T, _Dimension, _DefaultOrder, _IsPeriodic>::
         interpolate(argument_type x,
                     std::array<_T, _Dimension>* gradient) const
{
  // Ensure that the data for the grid array is valid.
#ifdef STLIB_DEBUG
  assert(_gridPointer);
  assert(_gridPointer->begin());
#endif
  // Calculate the scaled coordinates, in [0..1]^_Dimension, and the
  // determine the enclosing grid cell.
  IndexList cell;
  calculateScaledCoordinates<_Order>(&cell, &x);
  // Interpolate.
  const _T f = internal::interpolateFunction<_Order, _IsPeriodic>
               (*_gridPointer, cell, x, gradient);
  // Scale to go from logical coordinates to physical coordinates.
  *gradient *= _inverseWidths;
  return f;
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
typename InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder,
         _IsPeriodic>::result_type
         InterpolatingFunctionRegularGridBase
         <_T, _Dimension, _DefaultOrder, _IsPeriodic>::
         interpolate(const std::size_t order, const argument_type& x) const
{
  if (order == 0) {
    return interpolate<0>(x);
  }
  else if (order == 1) {
    return interpolate<1>(x);
  }
  else if (order == 3) {
    return interpolate<3>(x);
  }
  assert(false);
  return 0;
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
typename InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder,
         _IsPeriodic>::result_type
         InterpolatingFunctionRegularGridBase
         <_T, _Dimension, _DefaultOrder, _IsPeriodic>::
         interpolate(const std::size_t order, const argument_type& x,
                     std::array<_T, _Dimension>* gradient) const
{
  if (order == 0) {
    return interpolate<0>(x, gradient);
  }
  else if (order == 1) {
    return interpolate<1>(x, gradient);
  }
  else if (order == 3) {
    return interpolate<3>(x, gradient);
  }
  assert(false);
  return 0;
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
void
InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder, _IsPeriodic>::
calculateScaledCoordinates(IndexList* cell, argument_type* x,
                           std::false_type /*point-based*/) const
{
  // Convert to index coordinates.
  *x -= _lowerCorner;
  *x *= _inverseWidths;
  if (_IsPeriodic) {
    // Use periodicity to transform to the canonical domain.
    for (std::size_t i = 0; i != _Dimension; ++i) {
      (*x)[i] -= std::floor((*x)[i] * _inverseCellExtents[i])
                 * _cellExtents[i];
    }
  }
  // Determine the cell.
  *cell = ext::convert_array<Index>(*x);
  // Move to the nearest cell if the point is outside the grid. For the
  // periodic case this may occur due to roundoff error.
  for (std::size_t i = 0; i != _Dimension; ++i) {
    if ((*cell)[i] < 0) {
      (*cell)[i] = 0;
    }
    else if (Index(_cellExtents[i]) <= (*cell)[i]) {
      (*cell)[i] = _cellExtents[i] - 1;
    }
  }
  // Convert to the cell parametrization, which is [0..1)^_Dimension if the
  // point is inside the grid.
  for (std::size_t i = 0; i != x->size(); ++i) {
    (*x)[i] -= (*cell)[i];
  }
}


template<typename _T, std::size_t _Dimension, std::size_t _DefaultOrder,
         bool _IsPeriodic>
inline
void
InterpolatingFunctionRegularGridBase<_T, _Dimension, _DefaultOrder,
                                     _IsPeriodic>::
                                     calculateScaledCoordinates(IndexList* index, argument_type* x,
                                                                std::true_type /*point-based*/) const
{
  // Convert to index coordinates.
  (*x) -= _lowerCorner;
  (*x) *= _inverseWidths;
  // Add 0.5 to move from grid cells to grid points.
  (*x) += 0.5;
  if (_IsPeriodic) {
    // Use periodicity to transform to the canonical domain.
    for (std::size_t i = 0; i != _Dimension; ++i) {
      (*x)[i] -= std::floor((*x)[i] * _inverseCellExtents[i])
                 * _cellExtents[i];
    }
  }
  // Find the closest grid point by converting to an integer.
  Index gridExtent;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    (*index)[i] = Index((*x)[i]);
    gridExtent = _cellExtents[i] + 1 - _IsPeriodic;
    if ((*index)[i] < 0) {
      (*index)[i] = 0;
    }
    else if (gridExtent <= (*index)[i]) {
      (*index)[i] = gridExtent - 1;
    }
  }
}


} // namespace numerical
}

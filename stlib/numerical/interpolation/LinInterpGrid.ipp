// -*- C++ -*-

#if !defined(__numerical_interpolation_LinInterpGrid_ipp__)
#error This file is an implementation detail of LinInterpGrid.
#endif

namespace stlib
{
namespace numerical
{

namespace internal
{

template<typename F, typename T>
typename boost::call_traits<F>::param_type
lin_interp(const container::MultiArray<F, 1>& fields,
           typename container::MultiArray<F, 1>::IndexList i,
           const std::array<T, 1>& t)
{
  // Make these static in case F is not a fundamental type.
  static F f, g;
#ifdef STLIB_DEBUG
  assert(0 <= t[0] && t[0] <= 1);
#endif
  // The following is more efficient if F is not a fundamental type.

  f = fields(i);
  f *= (1 - t[0]);

  ++i[0];
  g = fields(i);
  g *= t[0];
  f += g;

  return f;
}

template<typename F, typename T>
typename boost::call_traits<F>::param_type
lin_interp(const container::MultiArray<F, 2>& fields,
           typename container::MultiArray<F, 2>::IndexList i,
           const std::array<T, 2>& t)
{
  // Make these static in case F is not a fundamental type.
  static F f, g;
#ifdef STLIB_DEBUG
  assert(0 <= t[0] && t[0] <= 1 && 0 <= t[1] && t[1] <= 1);
#endif
  // The following is more efficient if F is not a fundamental type.

  f = fields(i);
  f *= (1 - t[0]) * (1 - t[1]);

  ++i[0];
  g = fields(i);
  g *= t[0] * (1 - t[1]);
  f += g;
  --i[0];

  ++i[1];
  g = fields(i);
  g *= (1 - t[0]) * t[1];
  f += g;

  ++i[0];
  g = fields(i);
  g *= t[0] * t[1];
  f += g;

  return f;
}

template<typename F, typename T>
typename boost::call_traits<F>::param_type
lin_interp(const container::MultiArray<F, 3>& fields,
           typename container::MultiArray<F, 3>::IndexList i,
           const std::array<T, 3>& t)
{
  // Make these static in case F is not a fundamental type.
  static F f, g;
#ifdef STLIB_DEBUG
  assert(0 <= t[0] && t[0] <= 1 && 0 <= t[1] && t[1] <= 1 &&
         0 <= t[2] && t[2] <= 1);
#endif

  f = fields(i);
  f *= (1 - t[0]) * (1 - t[1]) * (1 - t[2]);

  ++i[0];
  g = fields(i);
  g *= t[0] * (1 - t[1]) * (1 - t[2]);
  f += g;
  --i[0];

  ++i[1];
  g = fields(i);
  g *= (1 - t[0]) * t[1] * (1 - t[2]);
  f += g;
  --i[1];

  ++i[0];
  ++i[1];
  g = fields(i);
  g *= t[0] * t[1] * (1 - t[2]);
  f += g;
  --i[0];
  --i[1];

  ++i[2];
  g = fields(i);
  g *= (1 - t[0]) * (1 - t[1]) * t[2];
  f += g;
  --i[2];

  ++i[0];
  ++i[2];
  g = fields(i);
  g *= t[0] * (1 - t[1]) * t[2];
  f += g;
  --i[0];
  --i[2];

  ++i[1];
  ++i[2];
  g = fields(i);
  g *= (1 - t[0]) * t[1] * t[2];
  f += g;
  --i[1];
  --i[2];

  ++i[0];
  ++i[1];
  ++i[2];
  g = fields(i);
  g *= t[0] * t[1] * t[2];
  f += g;
  --i[0];
  --i[1];
  --i[2];

  return f;
}

}

template<std::size_t N, typename F, typename T>
inline
typename LinInterpGrid<N, F, T>::result_type
LinInterpGrid<N, F, T>::
operator()(argument_type x) const
{
  static Point p, q;
  static IndexList i;

  //
  // Convert the Cartesian point to an index.
  //

  // Store the location.
  p = x;
  // Convert the location to a continuous index.
  _grid.convertLocationToIndex(&p);
  q = p;
  // Floor to get the integer index.
  for (std::size_t n = 0; n != N; ++n) {
    q[n] = std::floor(q[n]);
  }
  for (std::size_t n = 0; n != N; ++n) {
    i[n] = Index(q[n]);
  }
  // The scaled offsets.  Each coordinate is in the range [0..1).
  p -= q;

  // Check that the index is in the grid.
  for (std::size_t n = 0; n != N; ++n) {
    assert(0 <= i[n] && i[n] < Index(_grid.getExtents()[n]));
  }

  return internal::lin_interp(_fields, i, p);
}

} // namespace numerical
}

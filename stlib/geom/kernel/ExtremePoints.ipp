// -*- C++ -*-

#if !defined(__geom_ExtremePoints_ipp__)
#error This file is an implementation detail of the class ExtremePoints.
#endif

namespace stlib
{
namespace geom
{


template<typename _ExtremePoints>
inline
_ExtremePoints
extremePoints()
{
  using Number = typename _ExtremePoints::Number;
  _ExtremePoints x;
  for (std::size_t i = 0; i != _ExtremePoints::Dimension; ++i) {
    for (std::size_t j = 0; j != _ExtremePoints::Dimension; ++j) {
      x.points[i][0][j] = std::numeric_limits<Number>::quiet_NaN();
      x.points[i][1][j] = std::numeric_limits<Number>::quiet_NaN();
    }
  }
  return x;
}


template<typename _T, std::size_t _D>
inline
ExtremePoints<_T, _D>&
operator+=(ExtremePoints<_T, _D>& x, std::array<_T, _D> const& p)
{
  for (std::size_t i = 0; i != _D; ++i) {
    if (p[i] < x.points[i][0][i]) {
      x.points[i][0] = p;
    }
    if (p[i] > x.points[i][1][i]) {
      x.points[i][1] = p;
    }
  }
  return x;
}


template<typename _T, std::size_t _D>
inline
ExtremePoints<_T, _D>&
operator+=(ExtremePoints<_T, _D>& x, ExtremePoints<_T, _D> const& rhs)
{
  for (std::size_t i = 0; i != _D; ++i) {
    if (rhs.points[i][0][i] < x.points[i][0][i]) {
      x.points[i][0] = rhs.points[i][0];
    }
    if (rhs.points[i][1][i] > x.points[i][1][i]) {
      x.points[i][1] = rhs.points[i][1];
    }
  }
  return x;
}


template<typename _Float, typename _Float2, std::size_t _D>
struct ExtremePointsForObject<_Float, ExtremePoints<_Float2, _D> >
{
  std::size_t static constexpr Dimension = _D;
  using DefaultExtremePoints = ExtremePoints<_Float2, _D>;

  static
  ExtremePoints<_Float, _D>
  create(ExtremePoints<_Float2, _D> const& x)
  {
    ExtremePoints<_Float, _D> result;
    for (std::size_t i = 0; i != _D; ++i) {
      for (std::size_t j = 0; j != 2; ++j) {
        result.points[i][j] =
          ext::ConvertArray<_Float>::convert(x.points[i][j]);
      }
    }
    return result;
  }
};

/// Trivially convert an ExtremePoints to one of the same type.
/** This function differs from the one in which an ExtremePoints is converted
    to one with a different number type in that here we return a constant
    reference to the argument. Thus, we avoid constructing an ExtremePoints. */
template<typename _Float, std::size_t _D>
struct ExtremePointsForObject<_Float, ExtremePoints<_Float, _D> >
{
  std::size_t static constexpr Dimension = _D;
  using DefaultExtremePoints = ExtremePoints<_Float, _D>;

  static
  ExtremePoints<_Float, _D> const&
  create(ExtremePoints<_Float, _D> const& x)
  {
    return x;
  }
};

template<typename _Float, typename _Float2, std::size_t _D>
struct ExtremePointsForObject<_Float, std::array<_Float2, _D> >
{
  std::size_t static constexpr Dimension = _D;
  using DefaultExtremePoints = ExtremePoints<_Float2, _D>;

  static
  ExtremePoints<_Float, _D>
  create(std::array<_Float2, _D> const& x)
  {
    ExtremePoints<_Float, _D> result;
    auto const point = ext::ConvertArray<_Float>::convert(x);
    for (std::size_t i = 0; i != _D; ++i) {
      for (std::size_t j = 0; j != 2; ++j) {
        result.points[i][j] = point;
      }
    }
    return result;
  }
};

template<typename _Float, typename _Float2, std::size_t _D, std::size_t N>
struct ExtremePointsForObject<_Float, std::array<std::array<_Float2, _D>, N> >
{
  std::size_t static constexpr Dimension = _D;
  using DefaultExtremePoints = ExtremePoints<_Float2, _D>;

  static
  ExtremePoints<_Float, _D>
  create(std::array<std::array<_Float2, _D>, N> const& x)
  {
    static_assert(N != 0, "Error.");
    ExtremePoints<_Float, _D> result =
      extremePoints<ExtremePoints<_Float, _D> >(x[0]);
    for (std::size_t i = 1; i != N; ++i) {
      result += ext::ConvertArray<_Float>::convert(x[i]);
    }
    return result;
  }
};

template<typename _Float, typename _Geometric, typename _Data>
struct ExtremePointsForObject<_Float, std::pair<_Geometric, _Data> >
{
  std::size_t static constexpr Dimension =
    ExtremePointsForObject<_Float, _Geometric>::Dimension;
  using DefaultExtremePoints =
    typename ExtremePointsForObject<_Float, _Geometric>::DefaultExtremePoints;

  static
  ExtremePoints<_Float, Dimension>
  create(std::pair<_Geometric, _Data> const& x)
  {
    return extremePoints<ExtremePoints<_Float, Dimension> >(x.first);
  }
};


} // namespace geom
} // namespace stlib

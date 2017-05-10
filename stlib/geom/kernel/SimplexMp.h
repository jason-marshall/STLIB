// -*- C++ -*-

/**
  \file
  \brief An N-simplex where vertices 1..N-1 are represented as offsets from vertex 0.
*/

#if !defined(__stlib_geom_kernel_SimplexMp_h__)
#define __stlib_geom_kernel_SimplexMp_h__

#include "stlib/geom/kernel/ExtremePoints.h"
#include "stlib/geom/kernel/Simplex.h"

namespace stlib
{
namespace geom
{

/// An N-simplex where vertices 1..N-1 are represented as offsets from vertex 0.
/**
  \param _FloatCoord is the floating-point number type for coordinates.
  \param _FloatOffset is the floating-point number type for offsets.
  \param _D is the space dimension.
  \param N is the simplex dimension.

  This class is used to reduce the storage requirements for simplices
  by using mixed-precision.  Assume that one must represent simplex
  coordinates with double-precision, but only needs single-precision
  for calculated quantities, such as content or distance to the
  simplex. Then it is acceptable to store only the first vertex with
  double-precision numbers, and store the remaining vertices as
  offsets from the first with single-precision.
*/
template<typename _FloatCoord, typename _FloatOffset, std::size_t _D,
         std::size_t N>
struct SimplexMp {
  // Constants.

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t SpaceDimension = _D;
  /// The simplex dimension.
  BOOST_STATIC_CONSTEXPR std::size_t SimplexDimension = N;

  // Types.

  /// The number type for coordinates.
  using Float = _FloatCoord;
  /// The point type.
  using Point = std::array<Float, SpaceDimension>;

  // Member data

  /// The first vertex.
  Point vertex;
  /// Offsets for the remaining vertices.
  Simplex<_FloatOffset, SpaceDimension, SimplexDimension - 1> offsets;
};


/// Build a SimplexMp from the simplex.
/**
\note The first template parameter, the floating-point type for the
offsets must be specified explicitly.
*/
template<typename _FloatOffset, typename _FloatCoord, std::size_t _D,
         std::size_t Np1>
inline
SimplexMp<_FloatCoord, _FloatOffset, _D, Np1 - 1>
toMixedPrecision
(std::array<std::array<_FloatCoord, _D>, Np1> const& simplex)
{
  constexpr std::size_t N = Np1 - 1;
  SimplexMp<_FloatCoord, _FloatOffset, _D, N> result;
  result.vertex = simplex[0];
  for (std::size_t i = 0; i != N; ++i) {
    for (std::size_t j = 0; j != _D; ++j) {
      result.offsets[i][j] = simplex[i + 1][j] - simplex[0][j];
    }
  }
  return result;
}


/// Build a SimplexMp for each simplex.
/**
\note The first template parameter, the floating-point type for the
offsets must be specified explicitly.
*/
template<typename _FloatOffset, typename _FloatCoord, std::size_t _D,
         std::size_t Np1>
inline
std::vector<SimplexMp<_FloatCoord, _FloatOffset, _D, Np1 - 1> >
toMixedPrecision
(std::vector<std::array<std::array<_FloatCoord, _D>, Np1> > const& simplices)
{
  std::vector<SimplexMp<_FloatCoord, _FloatOffset, _D, Np1 - 1> >
    result(simplices.size());
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = toMixedPrecision<_FloatOffset>(simplices[i]);
  }
  return result;
}


/// Build a simplex from the SimplexMp.
template<typename _FloatCoord, typename _FloatOffset, std::size_t _D,
         std::size_t N>
inline
Simplex<_FloatCoord, _D, N>
toUniformPrecision(SimplexMp<_FloatCoord, _FloatOffset, _D, N> const& x)
{
  Simplex<_FloatCoord, _D, N> result;
  result[0] = x.vertex;
  for (std::size_t i = 0; i != N; ++i) {
    for (std::size_t j = 0; j != _D; ++j) {
      result[i + 1][j] = x.vertex[j] + x.offsets[i][j];
    }
  }
  return result;
}


/// Build a simplex from the SimplexMp.
template<typename _FloatCoord, typename _FloatOffset, std::size_t _D,
         std::size_t N>
inline
std::vector<Simplex<_FloatCoord, _D, N> >
toUniformPrecision
(std::vector<SimplexMp<_FloatCoord, _FloatOffset, _D, N> > const& x)
{
  std::vector<Simplex<_FloatCoord, _D, N> > result(x.size());
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = toUniformPrecision(x[i]);
  }
  return result;
}


//! Make a bounding box around the simplex.
/*! \relates BBox
  \relates SimplexMp */
template<typename _Float, typename _FloatCoord, typename _FloatOffset,
         std::size_t _D,std::size_t N>
struct BBoxForObject<_Float,
                     SimplexMp<_FloatCoord, _FloatOffset, _D, N> >
{
  using DefaultBBox = BBox<_FloatCoord, _D>;

  static
  BBox<_Float, _D>
  create(SimplexMp<_FloatCoord, _FloatOffset, _D, N> const& x)
  {
    // Convert to the regular representation of a simplex, and then build the
    // bounding box.
    return specificBBox<BBox<_Float, _D> >(toUniformPrecision(x));
  }
};


//! Make an extreme points bounding structure around the simplex.
/*! \relates ExtremePoints
  \relates SimplexMp */
template<typename _Float, typename _FloatCoord, typename _FloatOffset,
         std::size_t _D,std::size_t N>
struct ExtremePointsForObject
<_Float, SimplexMp<_FloatCoord, _FloatOffset, _D, N> >
{
  using DefaultExtremePoints = ExtremePoints<_FloatCoord, _D>;

  static
  ExtremePoints<_Float, _D>
  create(SimplexMp<_FloatCoord, _FloatOffset, _D, N> const& x)
  {
    // Convert to the regular representation of a simplex, and then build the
    // extreme points bounding structure.
    return extremePoints<ExtremePoints<_Float, _D> >(toUniformPrecision(x));
  }
};


} // namespace geom
} // namespace stlib

#endif

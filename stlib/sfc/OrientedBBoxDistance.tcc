// -*- C++ -*-

#if !defined(__sfc_OrientedBBoxDistance_tcc__)
#error This file is an implementation detail of OrientedBBoxDistance.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Float, std::size_t _D>
inline
OrientedBBoxDistance<_Float, _D>::
OrientedBBoxDistance(OrientedBBox const& orientedBBox) :
  _orientedBBox(orientedBBox),
  _corners(),
  _faceRadius2()
{
  // Ensure that the least significant direction is listed last.
  std::size_t const minIndex = std::min_element(_orientedBBox.radii.begin(),
                                                _orientedBBox.radii.end()) -
    _orientedBBox.radii.begin();
  if (_orientedBBox.radii[minIndex] < _orientedBBox.radii[Dimension - 1]) {
    std::swap(_orientedBBox.axes[minIndex],
              _orientedBBox.axes[Dimension - 1]);
    std::swap(_orientedBBox.radii[minIndex],
              _orientedBBox.radii[Dimension - 1]);
  }
  // Calculate the corners for the lower and upper faces.
  std::array<Point, Dimension> offsets(_orientedBBox.axes);
  for (std::size_t i = 0; i != Dimension; ++i) {
    offsets[i] *= _orientedBBox.radii[i];
  }
  for (std::size_t i = 0; i != NumFaceCorners; ++i) {
    Point p = _orientedBBox.center;
    for (std::size_t j = 0; j != Dimension - 1; ++j) {
      // Use the bits of i to derive directions.
      if (i >> j & 1) {
        p += offsets[j];
      }
      else {
        p -= offsets[j];
      }
    }
    _corners[0][i] = p - offsets[Dimension - 1];
    _corners[1][i] = p + offsets[Dimension - 1];
  }

  // Compute the squared radius of the largest face.
  _faceRadius2 = 0;
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    _faceRadius2 += _orientedBBox.radii[i] * _orientedBBox.radii[i];
  }
}


template<typename _Float, std::size_t _D>
inline
std::size_t
OrientedBBoxDistance<_Float, _D>::
getDirection(Point const& queryPointsCenter, _Float const queryPointsMaxRadius)
  const
{
  // Compute the signed distance to the plane that is orthogonal to the 
  // least significant principal direction.
  _Float const signedDistance =
    ext::dot(queryPointsCenter - _orientedBBox.center,
             _orientedBBox.axes[Dimension - 1]);
  _Float const threshold = _orientedBBox.radii[Dimension - 1] +
    queryPointsMaxRadius;
  if (signedDistance < - threshold ) {
    return 0;
  }
  else if (signedDistance > threshold ) {
    return 1;
  }
  return -1;
}


template<typename _Float, std::size_t _D>
inline
bool
OrientedBBoxDistance<_Float, _D>::
areAnyRelevant
(Point queryPointsCenter,
 _Float const queryPointsMaxRadius,
 std::size_t const direction, 
 std::vector<_Float, simd::allocator<_Float> > const& queryPointData,
 std::vector<_Float, simd::allocator<_Float> > const& upperBounds) const
{
#ifdef STLIB_DEBUG
  assert(direction == 0 || direction == 1);
  assert(queryPointData.size() == Dimension * upperBounds.size());
#endif

  // First determine for which corners we need to compute distance.
  std::array<bool, NumFaceCorners> useCorner =
    ext::filled_array<std::array<bool, NumFaceCorners> >(true);
  queryPointsCenter -= _orientedBBox.center;
  for (std::size_t i = 0; i != Dimension - 1; ++i) {
    _Float const signedDistance =
      ext::dot(queryPointsCenter, _orientedBBox.axes[i]);
    if (signedDistance > queryPointsMaxRadius) {
      std::size_t const mask = ~(1 << i);
      for (std::size_t j = 0; j != NumFaceCorners; ++j) {
        useCorner[j & mask] = false;
      }
    }
    else if (signedDistance < -queryPointsMaxRadius) {
      std::size_t const mask = 1 << i;
      for (std::size_t j = 0; j != NumFaceCorners; ++j) {
        useCorner[j | mask] = false;
      }
    }
  }
#ifdef STLIB_DEBUG
  assert(std::count(useCorner.begin(), useCorner.end(), true));
#endif

  Vector const negativeFaceRadius2 = simd::set1(-_faceRadius2);
  std::array<Vector, Dimension> cornerCoords;
  std::size_t const numBlocks = upperBounds.size() / VectorSize;
  // The outer loop is over the corner points. This is because it reduces
  // branches and use of set1().
  for (std::size_t i = 0; i != NumFaceCorners; ++i) {
    if (! useCorner[i]) {
      continue;
    }
    for (std::size_t j = 0; j != Dimension; ++j) {
      cornerCoords[j] = simd::set1(_corners[direction][i][j]);
    }
    _Float const* u = &upperBounds[0];
    _Float const* p = &queryPointData[0];
    // For each block of query points.
    for (std::size_t j = 0; j != numBlocks; ++j) {
      Vector lower = negativeFaceRadius2;
      for (std::size_t k = 0; k != Dimension; ++k) {
        Vector const d = simd::load(p) - cornerCoords[k];
        p += VectorSize;
        lower += d * d;
      }
      // A point is relevant if the lower bound is less than or equal to the 
      // upper bound.
      if (simd::moveMask(simd::lessEqual(lower, simd::load(u)))) {
        return true;
      }
      u += VectorSize;
    }
  }
  return false;
}


} // namespace sfc
}

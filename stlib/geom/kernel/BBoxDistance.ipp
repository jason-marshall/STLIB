// -*- C++ -*-

#if !defined(__geom_BBoxDistance_ipp__)
#error This file is an implementation detail of BBoxDistance.
#endif

namespace stlib
{
namespace geom
{


// Return the minimum distance between the two intervals.
template<typename _Float>
inline
_Float
minDist(const _Float aLower, const _Float aUpper,
        const _Float bLower, const _Float bUpper)
{
  // If a precedes b.
  if (aUpper < bLower) {
    return bLower - aUpper;
  }
  // If a follows b.
  else if (bUpper < aLower) {
    return aLower - bUpper;
  }
  // If the intervals overlap.
  return 0;
}


// Return the maximum distance between the two intervals.
template<typename _Float>
inline
_Float
maxDist(const _Float aLower, const _Float aUpper,
        const _Float bLower, const _Float bUpper)
{
  // The maximum distance must be between the upper bound of one interval
  // and the lower bound of the other.
  return std::max(aUpper - bLower, bUpper - aLower);
}


// Not used. This is not the most efficient.
#if 0
template<typename _Float>
inline
_Float
maxMinDist(const _Float aLower, const _Float aUpper,
           const _Float bLower, const _Float bUpper)
{
  _Float result = 0;
  const _Float bMid = 0.5 * (bLower + bUpper);
  // If the midpoint for b is in a, the midpoint is a potential maxima.
  if (aLower < bMid && bMid < aUpper) {
    result = bUpper - bMid;
  }
  // The lower point of a is a potential maxima.
  result = std::max(result,
                    std::min(std::abs(aLower - bLower),
                             std::abs(aLower - bUpper)));
  // The upper point of a is a potential maxima.
  result = std::max(result,
                    std::min(std::abs(aUpper - bLower),
                             std::abs(aUpper - bUpper)));
  return result;
}
#endif


// CONTINUE: This is not the most efficient.
#if 0
template<typename _Float>
inline
_Float
maxMinDist(const _Float aLower, const _Float aUpper,
           const _Float bLower, const _Float bUpper)
{
  const _Float bMid = 0.5 * (bLower + bUpper);
  // If the midpoint for b is in a, the midpoint is a potential maxima.
  if (aLower < bMid && bMid < aUpper) {
    _Float result = bUpper - bMid;
    // The lower point of a is a potential maxima. We don't need to check
    // the distance from aLower to bUpper.
    result = std::max(result, std::abs(aLower - bLower));
    // The upper point of a is a potential maxima. We don't need to check
    // the distance from aUpper to bLower.
    result = std::max(result, std::abs(aUpper - bUpper));
    return result;
  }
  else {
    // The lower and upper points of a are potential maxima.
    return std::max(std::min(std::abs(aLower - bLower),
                             std::abs(aLower - bUpper)),
                    std::min(std::abs(aUpper - bLower),
                             std::abs(aUpper - bUpper)));
  }
}
#endif


// Return the maximum minimum distance from a point in the first interval
// to an object whose tight bounding box is the second interval.
template<typename _Float>
inline
_Float
maxMinDist(const _Float aLower, const _Float aUpper,
           const _Float bLower, const _Float bUpper)
{
  const _Float bMid = 0.5 * (bLower + bUpper);
  // If the midpoint for b is in a, the midpoint is a potential maxima.
  if (aLower < bMid) {
    if (bMid < aUpper) {
      _Float result = bUpper - bMid;
      // The lower point of a is a potential maxima. We don't need to check
      // the distance from aLower to bUpper.
      result = std::max(result, std::abs(aLower - bLower));
      // The upper point of a is a potential maxima. We don't need to check
      // the distance from aUpper to bLower.
      result = std::max(result, std::abs(aUpper - bUpper));
      return result;
    }
    else {
      // aLower <= aUpper <= bMid
      // The lower and upper points of a are potential maxima.
      return std::max(std::abs(aLower - bLower),
                      std::abs(aUpper - bLower));
    }
  }
  else {
    // bMid <= aLower <= aUpper
    // The lower and upper points of a are potential maxima.
    return std::max(std::abs(aLower - bUpper),
                    std::abs(aUpper - bUpper));
  }
}


template<typename _Float, std::size_t _D>
inline
_Float
minDist2(const BBox<_Float, _D>& a, const std::array<_Float, _D>& p)
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    if (p[i] < a.lower[i]) {
      d2 += (a.lower[i] - p[i]) * (a.lower[i] - p[i]);
    }
    else if (a.upper[i] < p[i]) {
      d2 += (p[i] - a.upper[i]) * (p[i] - a.upper[i]);
    }
  }
  return d2;

}


template<typename _Float, std::size_t _D>
inline
_Float
minMinDist2(const BBox<_Float, _D>& a, const BBox<_Float, _D>& b)
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    // I could implement this in terms of minDist(),
    // d = minDist(...);
    // d2 += d * d;
    // but this is a bit more efficient.
    if (a.upper[i] < b.lower[i]) {
      d2 += (a.upper[i] - b.lower[i]) * (a.upper[i] - b.lower[i]);
    }
    else if (b.upper[i] < a.lower[i]) {
      d2 += (b.upper[i] - a.lower[i]) * (b.upper[i] - a.lower[i]);
    }
  }
  return d2;
}


// The maximum distance from a point in the BBox a to the point p.
template<typename _Float, std::size_t _D>
inline
_Float
maxDist2(const BBox<_Float, _D>& a, const std::array<_Float, _D>& p)
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    d2 += std::max((p[i] - a.lower[i]) * (p[i] - a.lower[i]),
                   (p[i] - a.upper[i]) * (p[i] - a.upper[i]));
  }
  return d2;
}


// This implementation is slower.
#if 0
template<typename _Float, std::size_t _D>
inline
_Float
maxDist2(const BBox<_Float, _D>& a, const std::array<_Float, _D>& p)
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    if (p[i] > 0.5 * (a.lower[i] + a.upper[i])) {
      d2 += (p[i] - a.lower[i]) * (p[i] - a.lower[i]);
    }
    else {
      d2 += (p[i] - a.upper[i]) * (p[i] - a.upper[i]);
    }
  }
  return d2;
}
#endif


// The maximum distance between any point in the first box and any point in
// the second box.
template<typename _Float, std::size_t _D>
inline
_Float
maxMaxDist2(const BBox<_Float, _D>& a, const BBox<_Float, _D>& b)
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    const _Float d = maxDist(a.lower[i], a.upper[i], b.lower[i], b.upper[i]);
    d2 += d * d;
  }
  return d2;
}


template<typename _Float, std::size_t _D>
inline
_Float
nxnDist2(const BBox<_Float, _D>& a, const BBox<_Float, _D>& b)
{
  _Float maxMaxD2 = 0;
  std::array<_Float, _D> maxD2;
  for (std::size_t i = 0; i != _D; ++i) {
    _Float d = maxDist(a.lower[i], a.upper[i], b.lower[i], b.upper[i]);
    d = d * d;
    maxD2[i] = d;
    maxMaxD2 += d;
  }
  _Float nxnD2 = maxMaxD2;
  for (std::size_t i = 0; i != _D; ++i) {
    _Float maxMinD2 = maxMinDist(a.lower[i], a.upper[i],
                                 b.lower[i], b.upper[i]);
    maxMinD2 = maxMinD2 * maxMinD2;
    nxnD2 = std::min(nxnD2, maxMaxD2 - maxD2[i] + maxMinD2);
  }
  return nxnD2;
}


template<typename _Float, std::size_t _D, std::size_t N>
inline
_Float
upperBoundDist2(const BBox<_Float, _D>& a,
                const std::array<std::array<_Float, _D>, N>& points)
{
  _Float bound = maxDist2(a, points[0]);
  for (std::size_t i = 1; i != N; ++i) {
    bound = std::min(bound, maxDist2(a, points[i]));
  }
  return bound;
}

} // namespace geom
}

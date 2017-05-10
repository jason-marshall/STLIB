// -*- C++ -*-

#if !defined(__geom_BBoxDistanceSimd_ipp__)
#error This file is an implementation detail of BBoxDistanceSimd.
#endif

namespace stlib
{
namespace geom
{


template<typename _Float, std::size_t _D>
inline
BBoxDistance<_Float, _D>::
BBoxDistance(BBox const& box)
{
  for (std::size_t i = 0; i != Dimension; ++i) {
    _lower[i] = simd::set1(box.lower[i]);
    _upper[i] = simd::set1(box.upper[i]);
  }
}


template<typename _Float, std::size_t _D>
inline
void
BBoxDistance<_Float, _D>::
lowerBound2(AlignedVector const& lowerData,
            AlignedVector const& upperData,
            AlignedVector* lowerBounds) const
{
  assert(lowerData.size() == upperData.size());
  assert(lowerData.size() % (Dimension * VectorSize) == 0);
  lowerBounds->resize(lowerData.size() / Dimension);
  Float const* loData = &lowerData[0];
  Float const* upData = &upperData[0];
  Float* boundsData = &(*lowerBounds)[0];
  // For each block of boxes.
  const std::size_t numBlocks = lowerBounds->size() / VectorSize;
  for (std::size_t i = 0; i != numBlocks; ++i) {
    Vector d2 = simd::setzero<Float>();
    // Process the coordinates from each dimension.
    for (std::size_t j = 0; j != Dimension; ++j) {
      // The first is nonzero if this box precedes the other.
      // The second is nonzero if this box follows the other.
      Vector const d = simd::max(simd::load(loData) - _upper[j],
                                 simd::setzero<Float>()) +
        simd::max(_lower[j] - simd::load(upData),
                  simd::setzero<Float>());
      d2 += d * d;
      loData += VectorSize;
      upData += VectorSize;
    }
    simd::store(boundsData, d2);
    boundsData += VectorSize;
  }
}


template<typename _Float, std::size_t _D>
inline
typename BBoxDistance<_Float, _D>::Vector
BBoxDistance<_Float, _D>::
_maxDist2(const Float* soaData) const
{
  Vector d2 = simd::setzero<Float>();
  for (std::size_t i = 0; i != Dimension; ++i) {
    Vector p = simd::load(soaData);
    soaData += VectorSize;
    d2 += simd::max((p - _lower[i]) * (p - _lower[i]),
                    (p - _upper[i]) * (p - _upper[i]));
  }
  return d2;
}


template<typename _Float, std::size_t _D>
inline
typename BBoxDistance<_Float, _D>::Vector
BBoxDistance<_Float, _D>::
_maxDist2(AlignedCoordinates const& points,
          std::size_t const blockIndex) const
{
#ifdef STLIB_DEBUG
  assert(blockIndex % VectorSize == 0);
#endif
  Vector p = simd::load(&points[0][blockIndex]);
  Vector d2 = simd::max((p - _lower[0]) * (p - _lower[0]),
                        (p - _upper[0]) * (p - _upper[0]));
  for (std::size_t i = 1; i != Dimension; ++i) {
    p = simd::load(&points[i][blockIndex]);
    d2 += simd::max((p - _lower[i]) * (p - _lower[i]),
                    (p - _upper[i]) * (p - _upper[i]));
  }
  return d2;
}


template<typename _Float, std::size_t _D>
inline
typename BBoxDistance<_Float, _D>::Vector
BBoxDistance<_Float, _D>::
_lowerBound2(AlignedCoordinates const& lower,
             AlignedCoordinates const& upper,
             std::size_t const blockIndex) const
{
#ifdef STLIB_DEBUG
  assert(blockIndex % VectorSize == 0);
#endif
  Vector d2 = simd::setzero<Float>();
  for (std::size_t j = 0; j != Dimension; ++j) {
    // The first is nonzero if this box precedes the other.
    // The second is nonzero if this box follows the other.
    Vector const d = simd::max(simd::load(&lower[j][blockIndex]) - _upper[j],
                               simd::setzero<Float>()) +
      simd::max(_lower[j] - simd::load(&upper[j][blockIndex]),
                simd::setzero<Float>());
    d2 += d * d;
  }
  return d2;
}


template<typename _Float, std::size_t _D>
template<std::size_t _PtsPerBox>
inline
void
BBoxDistance<_Float, _D>::
upperBound2(std::array<AlignedVector, _PtsPerBox> const& extremePointData,
            std::vector<Float, simd::allocator<Float>>* upperBounds) const
{
  assert(extremePointData[0].size() % (Dimension * VectorSize) == 0);
  for (std::size_t i = 1; i != _PtsPerBox; ++i) {
    assert(extremePointData[i].size() == extremePointData[0].size());
  }

  upperBounds->resize(extremePointData[0].size() / Dimension);
  std::array<Float const*, _PtsPerBox> ptData;
  for (std::size_t i = 0; i != ptData.size(); ++i) {
    ptData[i] = &extremePointData[i][0];
  }
  Float* boundsData = &(*upperBounds)[0];
  // For each block of points.
  const std::size_t numBlocks = upperBounds->size() / VectorSize;
  for (std::size_t i = 0; i != numBlocks; ++i) {
    Vector bound = _maxDist2(ptData[0]);
    for (std::size_t j = 1; j != _PtsPerBox; ++j) {
      bound = simd::min(bound, _maxDist2(ptData[j]));
    }
    simd::store(boundsData, bound);
    for (std::size_t j = 0; j != ptData.size(); ++j) {
      ptData[j] += Dimension * VectorSize;
    }
    boundsData += VectorSize;
  }
}


template<typename _Float, std::size_t _D>
inline
typename BBoxDistance<_Float, _D>::Float
BBoxDistance<_Float, _D>::
upperBound2(AlignedVector const& pointData) const
{
  assert(pointData.size() % (Dimension * VectorSize) == 0);
  Float const* ptData = &pointData[0];
  Vector bound = simd::set1(std::numeric_limits<Float>::infinity());
  // For each block of points.
  std::size_t const Block = Dimension * VectorSize;
  std::size_t const numBlocks = pointData.size() / Block;
  for (std::size_t i = 0; i != numBlocks; ++i) {
    bound = simd::min(bound, _maxDist2(ptData));
    ptData += Block;
  }
  ALIGN_SIMD Float b[VectorSize];
  simd::store(b, bound);
  return *std::min_element(b, b + VectorSize);
}


template<typename _Float, std::size_t _D>
inline
void
BBoxDistance<_Float, _D>::
lowerLessEqualUpper2(AlignedCoordinates const& lower,
                     AlignedCoordinates const& upper,
                     AlignedCoordinates const& points,
                     std::vector<unsigned char>* relevantObjects) const
{
  lowerLessEqualUpper2(_upperBound2(points), lower, upper, relevantObjects);
}


template<typename _Float, std::size_t _D>
inline
void
BBoxDistance<_Float, _D>::
lowerLessEqualUpper2(_Float const upperBound_,
                     AlignedCoordinates const& lower,
                     AlignedCoordinates const& upper,
                     std::vector<unsigned char>* relevantObjects) const
{
#ifdef STLIB_DEBUG
  assert(! lower[0].empty());
  assert(lower[0].size() % VectorSize == 0);
  for (std::size_t i = 1; i != Dimension; ++i) {
    assert(lower[i].size() == lower[0].size());
  }
  for (std::size_t i = 0; i != Dimension; ++i) {
    assert(upper[i].size() == lower[0].size());
  }
#endif
  relevantObjects->resize(lower[0].size());
  // First calculate an upper bound on the distance using the points.
  Vector const upperBound = simd::set1(upperBound_);
  ALIGN_SIMD Float lessEqual[VectorSize];
  std::size_t n = 0;
  // Loop over the objects a block at a time.
  for (std::size_t i = 0; i != lower[0].size(); i += VectorSize) {
    simd::store(lessEqual, simd::lessEqual(_lowerBound2(lower, upper, i),
                                           upperBound));
    for (std::size_t j = 0; j != VectorSize; ++j) {
      (*relevantObjects)[n++] = *reinterpret_cast<unsigned char const*>
        (&lessEqual[j]) & 1;
    }
  }
  // Mark the padded positions, which have NaN's for the lower and upper bounds,
  // as not relevant. This is necessary because the lower bounds for such 
  // positions will be calculated as 0.
  // Loop over the last block. (Only the last block may contain NaN's.)
  for (std::size_t i = lower[0].size() - VectorSize; i != lower[0].size();
       ++i) {
    if (lower[0][i] != lower[0][i]) {
      (*relevantObjects)[i] = 0;
    }
  }
#ifdef STLIB_DEBUG
  unsigned char anySet = 0;
  for (std::size_t i = 0; i != relevantObjects->size(); ++i) {
    anySet |= (*relevantObjects)[i];
  }
  assert(anySet);
#endif
}


template<typename _Float, std::size_t _D>
inline
typename BBoxDistance<_Float, _D>::Float
BBoxDistance<_Float, _D>::
_upperBound2(AlignedCoordinates const& points) const
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != Dimension; ++i) {
    assert(points[i].size() % VectorSize == 0);
  }
#endif
  Vector bound = simd::set1(std::numeric_limits<Float>::infinity());
  for (std::size_t i = 0; i != points[0].size(); i += VectorSize) {
    bound = simd::min(bound, _maxDist2(points, i));
  }
  ALIGN_SIMD Float b[VectorSize];
  simd::store(b, bound);
  return *std::min_element(b, b + VectorSize);
}


} // namespace geom
}

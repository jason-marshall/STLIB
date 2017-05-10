// -*- C++ -*-

#if !defined(__levelSet_PatchDistanceIdentifier_ipp__)
#error This file is an implementation detail of PatchDistanceIdentifier.
#endif

namespace stlib
{
namespace levelSet
{


#ifdef STLIB_NO_SIMD_INTRINSICS
//----------------------------------------------------------------------------
// Scalar.
//----------------------------------------------------------------------------


// Set the lower corner. Initialize the grid values.
inline
void
PatchDistanceIdentifier::
initialize(const Point& lowerCorner, const float value)
{
  Base::initialize(lowerCorner, value);
  std::fill(identifiers.begin(), identifiers.end(),
            std::numeric_limits<unsigned>::max());
}


// Add the ball with a union operation.
inline
void
PatchDistanceIdentifier::
unionEuclidean(const Ball& ball, const unsigned id)
{
  // Compute the squared distance components from the ball center.
  computeDistanceComponents(ball.center);
  // Update the Euclidean distance at each grid point.
  float d;
  std::size_t ko, offset;
  for (std::size_t k = 0; k != Extent; ++k) {
    ko = k * Extent * Extent;
    for (std::size_t j = 0; j != Extent; ++j) {
      offset = ko + j * Extent;
      for (std::size_t i = 0; i != Extent; ++i) {
        d = std::sqrt(_dx[i] + _dy[j] + _dz[k]) - ball.radius;
        if (d < grid[offset + i]) {
          grid[offset + i] = d;
          identifiers[offset + i] = id;
        }
      }
    }
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistanceIdentifier::
conditionalSetValueGe(const float threshold, const float value,
                      const unsigned id)
{
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (grid[i] >= threshold) {
      grid[i] = value;
      identifiers[i] = id;
    }
  }
}


#elif defined(__AVX2__)
//----------------------------------------------------------------------------
// AVX2
//----------------------------------------------------------------------------


// Set the lower corner. Initialize the grid values.
inline
void
PatchDistanceIdentifier::
initialize(const Point& lowerCorner, const float value)
{
  Base::initialize(lowerCorner, value);
  identifiers.fill(_mm256_set1_epi32(std::numeric_limits<unsigned>::max()));
}


// Add the ball with a union operation.
inline
void
PatchDistanceIdentifier::
unionEuclidean(const Ball& ball, const unsigned id)
{
  // Compute the squared distance components from the ball center.
  computeDistanceComponents(ball.center);
  // Update the Euclidean distance at each grid point.
  const Vector r = _mm256_set1_ps(ball.radius);
  const __m256i iv = _mm256_set1_epi32(id);
  __m256 d, comp;
  for (std::size_t i = 0; i != NumVectors; ++i) {
    d = _mm256_sqrt_ps(_dx[i & 0x3] + _dy[(i >> 2) & 0x3] + _dz[i >> 4]) - r;
    // If the new distance is less than the old distance.
    comp = _mm256_cmp_ps(d, grid[i], 1 /*<*/);
    grid[i] = simd::conditional(comp, d, grid[i]);
    identifiers[i] = simd::conditional(_mm256_castps_si256(comp), iv,
                                       identifiers[i]);
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistanceIdentifier::
conditionalSetValueGe(const float threshold, const float value,
                      const unsigned id)
{
  const __m256 t = _mm256_set1_ps(threshold);
  const __m256 v = _mm256_set1_ps(value);
  const __m256i iv = _mm256_set1_epi32(id);
  __m256 comp;
  for (std::size_t i = 0; i != grid.size(); ++i) {
    // Greater than or equal to the threshold.
    comp = _mm256_cmp_ps(grid[i], t, 13 /*>=*/);
    grid[i] = simd::conditional(comp, v, grid[i]);
    identifiers[i] = simd::conditional(_mm256_castps_si256(comp), iv,
                                       identifiers[i]);
  }
}


#else // __SSE__
//----------------------------------------------------------------------------
// SSE
//----------------------------------------------------------------------------


// Set the lower corner. Initialize the grid values.
inline
void
PatchDistanceIdentifier::
initialize(const Point& lowerCorner, const float value)
{
  Base::initialize(lowerCorner, value);
  std::fill(identifiers.begin(), identifiers.end(),
            _mm_set1_epi32(std::numeric_limits<unsigned>::max()));
}


// Add the ball with a union operation.
inline
void
PatchDistanceIdentifier::
unionEuclidean(const Ball& ball, const unsigned id)
{
  // Compute the squared distance components from the ball center.
  computeDistanceComponents(ball.center);
  // Update the Euclidean distance at each grid point.
  const Vector r = _mm_set1_ps(ball.radius);
  const __m128i iv = _mm_set1_epi32(id);
  __m128 d, comp;
  __m128i compi;
  for (std::size_t i = 0; i != NumVectors; ++i) {
    d = _mm_sqrt_ps(_dx[i & 0x3] + _dy[(i >> 2) & 0x3] + _dz[i >> 4]) - r;
    // If the new distance is less than the old distance.
    comp = _mm_cmplt_ps(d, grid[i]);
    grid[i] = simd::conditional(comp, d, grid[i]);
    compi = _mm_load_si128(reinterpret_cast<__m128i*>(&comp));
    identifiers[i] = simd::conditional(compi, iv, identifiers[i]);
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistanceIdentifier::
conditionalSetValueGe(const float threshold, const float value,
                      const unsigned id)
{
  const __m128 t = _mm_set1_ps(threshold);
  const __m128 v = _mm_set1_ps(value);
  const __m128i iv = _mm_set1_epi32(id);
  __m128 comp;
  __m128i compi;
  for (std::size_t i = 0; i != grid.size(); ++i) {
    // Greater than or equal to the threshold.
    comp = _mm_cmpge_ps(grid[i], t);
    grid[i] = simd::conditional(comp, v, grid[i]);
    compi = _mm_load_si128(reinterpret_cast<__m128i*>(&comp));
    identifiers[i] = simd::conditional(compi, iv, identifiers[i]);
  }
}


#endif

} // namespace levelSet
}

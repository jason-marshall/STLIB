// -*- C++ -*-

#if !defined(__levelSet_PatchDistance_ipp__)
#error This file is an implementation detail of PatchDistance.
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
PatchDistance::
initialize(const Point& lowerCorner, const float value)
{
  _lowerCorner = lowerCorner;
  std::fill(grid.begin(), grid.end(), value);
}


// Add the ball with a union operation.
inline
void
PatchDistance::
unionEuclidean(const Ball& ball)
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
        }
      }
    }
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistance::
conditionalSetValueGe(const float threshold, const float value)
{
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (grid[i] >= threshold) {
      grid[i] = value;
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
PatchDistance::
initialize(const Point& lowerCorner, const float value)
{
  _lowerCorner = lowerCorner;
  grid.fill(_mm256_set1_ps(value));
}


// Add the ball with a union operation.
inline
void
PatchDistance::
unionEuclidean(const Ball& ball)
{
  // Compute the squared distance components from the ball center.
  computeDistanceComponents(ball.center);
  // Update the Euclidean distance at each grid point.
  const Vector r = _mm256_set1_ps(ball.radius);
  for (std::size_t i = 0; i != NumVectors; ++i) {
    grid[i] =
      _mm256_min_ps(grid[i],
                    _mm256_sqrt_ps(_dx[i & 0x3] + _dy[(i >> 2) & 0x3] +
                                   _dz[i >> 4]) - r);
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistance::
conditionalSetValueGe(const float threshold, const float value)
{
  const __m256 t = _mm256_set1_ps(threshold);
  const __m256 v = _mm256_set1_ps(value);
  for (std::size_t i = 0; i != grid.size(); ++i) {
    // Greater than or equal to the threshold.
    grid[i] = simd::conditional(_mm256_cmp_ps(grid[i], t, 13 /*>=*/), v,
                                grid[i]);
  }
}


#else // __SSE__
//----------------------------------------------------------------------------
// SSE
//----------------------------------------------------------------------------


// Set the lower corner. Initialize the grid values.
inline
void
PatchDistance::
initialize(const Point& lowerCorner, const float value)
{
  _lowerCorner = lowerCorner;
  std::fill(grid.begin(), grid.end(), _mm_set1_ps(value));
}


// Add the ball with a union operation.
inline
void
PatchDistance::
unionEuclidean(const Ball& ball)
{
  // Compute the squared distance components from the ball center.
  computeDistanceComponents(ball.center);
  // Update the Euclidean distance at each grid point.
  const Vector r = _mm_set1_ps(ball.radius);
  for (std::size_t i = 0; i != NumVectors; ++i) {
    grid[i] =
      _mm_min_ps(grid[i],
                 _mm_sqrt_ps(_dx[i & 0x3] + _dy[(i >> 2) & 0x3] +
                             _dz[i >> 4]) - r);
  }
}


// For all grid points that are greater than or equal to the threshold,
// set to the specified value.
inline
void
PatchDistance::
conditionalSetValueGe(const float threshold, const float value)
{
  const __m128 t = _mm_set1_ps(threshold);
  const __m128 v = _mm_set1_ps(value);
  for (std::size_t i = 0; i != grid.size(); ++i) {
    // Greater than or equal to the threshold.
    grid[i] = simd::conditional(_mm_cmpge_ps(grid[i], t), v, grid[i]);
  }
}


#endif

} // namespace levelSet
}

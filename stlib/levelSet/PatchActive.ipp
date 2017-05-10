// -*- C++ -*-

#if !defined(__levelSet_PatchActive_ipp__)
#error This file is an implementation detail of PatchActive.
#endif

namespace stlib
{
namespace levelSet
{


#ifdef STLIB_NO_SIMD_INTRINSICS
//----------------------------------------------------------------------------
// Scalar.
//----------------------------------------------------------------------------


// Set the lower corner. Make all points active or inactive.
inline
void
PatchActive::
initialize(const Point& lowerCorner, const bool areActive)
{
  _lowerCorner = lowerCorner;
  if (areActive) {
    // Make all points active.
    for (std::size_t i = 0; i != NumPoints; ++i) {
      activeIndices[i] = i;
    }
    numActive = NumPoints;
  }
  else {
    numActive = 0;
  }
}


inline
void
PatchActive::
initializePositive(const Point& lowerCorner,
                   const std::array<Vector, NumVectors>& grid)
{
  _lowerCorner = lowerCorner;
  numActive = 0;
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (grid[i] > 0) {
      activeIndices[numActive++] = i;
    }
  }
}


inline
void
PatchActive::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by setting the index to NumVectors,
  // an invalid value.
  const float r2 = ball.radius * ball.radius;
  for (std::size_t i = 0; i != numActive; ++i) {
    const unsigned n = activeIndices[i];
    // 1 2  4 8  16 32  64 128  256 512
    // 1 1  1
    //        1  1  1
    //                  1  1    1
    if (_dx[n & 0x7] + _dy[(n >> 3) & 0x7] + _dz[n >> 6] < r2) {
      activeIndices[i] = NumVectors;
    }
  }
  // Move the inactive points.
  moveInactive();
}


#elif defined(__AVX2__)
//----------------------------------------------------------------------------
// AVX2
//----------------------------------------------------------------------------


inline
void
PatchActive::
initializePositive(const Point& lowerCorner,
                   const std::array<Vector, NumVectors>& grid)
{
  _lowerCorner = lowerCorner;
  numActive = 0;
  const __m256 zero = _mm256_setzero_ps();
  for (std::size_t i = 0; i != grid.size(); ++i) {
    activeIndices[numActive] = i;
    activeMasks[numActive] = _mm256_movemask_ps(_mm256_cmp_ps(grid[i], zero,
                             14 /*>*/));
    // If any of the elements of the SIMD vector are active.
    if (activeMasks[numActive]) {
      ++numActive;
    }
  }
}


inline
void
PatchActive::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by updating the mask.
  const Vector r2 = _mm256_set1_ps(ball.radius * ball.radius);
  for (std::size_t i = 0; i != numActive; ++i) {
    const unsigned n = activeIndices[i];
    // 1 2  4 8  16 32
    // 1 1
    //      1 1
    //           1  1
    activeMasks[i] &= _mm256_movemask_ps
                      (_mm256_cmp_ps(r2, _dx[n & 0x3] + _dy[(n >> 2) & 0x3] +
                                     _dz[n >> 4], 1/*<*/));
  }
  // Move the inactive vectors.
  moveInactive();
}


#else
//----------------------------------------------------------------------------
// SSE
//----------------------------------------------------------------------------


inline
void
PatchActive::
initializePositive(const Point& lowerCorner,
                   const std::array<Vector, NumVectors>& grid)
{
  _lowerCorner = lowerCorner;
  numActive = 0;
  const __m128 zero = _mm_setzero_ps();
  for (std::size_t i = 0; i != grid.size(); ++i) {
    activeIndices[numActive] = i;
    activeMasks[numActive] = _mm_movemask_ps(_mm_cmpgt_ps(grid[i], zero));
    // If any of the elements of the SIMD vector are active.
    if (activeMasks[numActive]) {
      ++numActive;
    }
  }
}


inline
void
PatchActive::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by updating the mask.
  const Vector r2 = _mm_set1_ps(ball.radius * ball.radius);
  for (std::size_t i = 0; i != numActive; ++i) {
    const unsigned n = activeIndices[i];
    // 1 2  4 8  16 32  64 128  256 512
    // 1 1
    //      1 1
    //           1  1   1
    activeMasks[i] &= _mm_movemask_ps
                      (_mm_cmplt_ps(r2, _dx[n & 0x3] + _dy[(n >> 2) & 0x3] + _dz[n >> 4]));
  }
  // Move the inactive vectors.
  moveInactive();
}




#endif


#ifdef STLIB_NO_SIMD_INTRINSICS
//----------------------------------------------------------------------------
// Scalar.
//----------------------------------------------------------------------------


inline
std::size_t
PatchActive::
numActivePoints()
{
  return numActive;
}


// Move the inactive points out of the active region.
inline
void
PatchActive::
moveInactive()
{
  std::size_t i = 0;
  while (i != numActive) {
    // If the point is inactive.
    if (activeIndices[i] == NumVectors) {
      // Move it to the inactive range.
      --numActive;
      activeIndices[i] = activeIndices[numActive];
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
}


#else
//----------------------------------------------------------------------------
// AVX2 or SSE
//----------------------------------------------------------------------------


// Set the lower corner. Make all points active or inactive.
inline
void
PatchActive::
initialize(const Point& lowerCorner, const bool areActive)
{
  _lowerCorner = lowerCorner;
  if (areActive) {
    // Make all points active.
    for (std::size_t i = 0; i != NumVectors; ++i) {
      activeIndices[i] = i;
#ifdef __AVX2__
      activeMasks[i] = 0xFF;
#else
      activeMasks[i] = 0xF;
#endif
    }
    numActive = NumVectors;
  }
  else {
    numActive = 0;
  }
}


// Return the number of active grid points.
inline
std::size_t
PatchActive::
numActivePoints()
{
  const std::size_t N = sizeof(std::size_t);
  // First clean up the masks by zeroing some of the inactive ones. This
  // enables us to use more efficient, 64-bit operations.
  const std::size_t size = (numActive + N - 1) / N;
  memset(&activeMasks[numActive], 0, N * size - numActive);
  // Count the active elements among the active SIMD vectors.
  std::size_t count = 0;
  const std::size_t* masks =
    reinterpret_cast<const std::size_t*>(&activeMasks[0]);
  for (std::size_t i = 0; i != size; ++i) {
    count += numerical::popCount(masks[i]);
  }
  return count;
}


// Move the inactive points out of the active region.
inline
void
PatchActive::
moveInactive()
{
  std::size_t i = 0;
  while (i != numActive) {
    // If each of the the points in the SIMD vector are inactive.
    if (activeMasks[i] == 0) {
      // Move it to the inactive range.
      --numActive;
      activeIndices[i] = activeIndices[numActive];
      activeMasks[i] = activeMasks[numActive];
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
}


#endif


} // namespace levelSet
}

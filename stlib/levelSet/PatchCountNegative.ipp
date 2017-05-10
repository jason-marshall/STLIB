// -*- C++ -*-

#if !defined(__levelSet_PatchCountNegative_ipp__)
#error This file is an implementation detail of PatchCountNegative.
#endif

namespace stlib
{
namespace levelSet
{


#ifdef STLIB_NO_SIMD_INTRINSICS
//----------------------------------------------------------------------------
// Scalar.
//----------------------------------------------------------------------------


// Set the lower corner. Make all points active.
inline
void
PatchCountNegative::
initialize(const Point& lowerCorner)
{
  _lowerCorner = lowerCorner;
  // Make all points active.
  for (std::size_t i = 0; i != NumVectors; ++i) {
    _activeIndices[i] = i;
  }
  _numActive = NumVectors;
}


inline
void
PatchCountNegative::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (_numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by setting the index to NumVectors,
  // an invalid value.
  const float r2 = ball.radius * ball.radius;
  for (std::size_t i = 0; i != _numActive; ++i) {
    const unsigned n = _activeIndices[i];
    // 1 2  4 8  16 32  64 128  256 512
    // 1 1  1
    //        1  1  1
    //                  1  1    1
    if (_dx[n & 0x7] + _dy[(n >> 3) & 0x7] + _dz[n >> 6] < r2) {
      _activeIndices[i] = NumVectors;
    }
  }
  // Move the invalid points.
  std::size_t i = 0;
  while (i != _numActive) {
    // If the point is inside the ball.
    if (_activeIndices[i] == NumVectors) {
      // Move it to the inactive range.
      --_numActive;
      _activeIndices[i] = _activeIndices[_numActive];
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
}


inline
std::size_t
PatchCountNegative::
numNegative() const
{
  return NumPoints - _numActive;
}


#elif defined(__AVX2__)
//----------------------------------------------------------------------------
// AVX2
//----------------------------------------------------------------------------


// Set the lower corner. Make all points active.
inline
void
PatchCountNegative::
initialize(const Point& lowerCorner)
{
  _lowerCorner = lowerCorner;
  // Make all points active.
  for (std::size_t i = 0; i != NumVectors; ++i) {
    _activeIndices[i] = i;
    _activeMasks[i] = 0xFF;
  }
  _numActive = NumVectors;
}


inline
void
PatchCountNegative::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (_numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by updating the mask.
  const Vector = _mm256_set1_ps(ball.radius * ball.radius);
  for (std::size_t i = 0; i != _numActive; ++i) {
    const unsigned n = _activeIndices[i];
    // 1 2  4 8  16 32
    // 1 1
    //      1 1
    //           1  1
    _activeMasks[i] &= _mm256_movemask_ps
                       (_mm256_cmp_ps(r2, _mm256_add_ps(_mm256_add_ps(_dx[n & 0x3],
                                      _dy[(n >> 2) & 0x3]),
                                      _dz[n >> 4]), 1/*<*/));
  }
  // Move the invalid points.
  std::size_t i = 0;
  while (i != _numActive) {
    // If each of the the points in the SIMD vector are inside the ball.
    if (_activeMasks[i] == 0) {
      // Move it to the inactive range.
      --_numActive;
      _activeIndices[i] = _activeIndices[_numActive];
      _activeMasks[i] = _activeMasks[_numActive];
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
}


// Return the number of grid points with negative distance.
inline
std::size_t
PatchCountNegative::
numNegative() const
{
  // The number of one bits in the lower or upper four bits.
  const unsigned char CountBits[] = {
    0, 1, 1, 2,
    1, 2, 2, 3,
    1, 2, 2, 3,
    2, 3, 3, 4
  };

  // First count the active (non-negative) elements among the active SIMD
  // vectors.
  std::size_t count = 0;
  for (std::size_t i = 0; i != _numActive; ++i) {
    count += CountBits[_activeMasks[i] & 0xF];
    count += CountBits[_activeMasks[i] >> 4];
  }
  return NumPoints - count;
}


#else
//----------------------------------------------------------------------------
// SSE
//----------------------------------------------------------------------------


// Set the lower corner. Make all points active.
inline
void
PatchCountNegative::
initialize(const Point& lowerCorner)
{
  _lowerCorner = lowerCorner;
  // Make all points active.
  for (std::size_t i = 0; i != NumVectors; ++i) {
    _activeIndices[i] = i;
    _activeMasks[i] = 0xF;
  }
  _numActive = NumVectors;
}


inline
void
PatchCountNegative::
clip(const Ball& ball)
{
  // If there are no active points, do nothing.
  if (_numActive == 0) {
    return;
  }
  // Compute the components of the squared distance.
  computeDistanceComponents(ball.center);
  // Mark the points with negative distance by updating the mask.
  const Vector r2 = _mm_set1_ps(ball.radius * ball.radius);
  for (std::size_t i = 0; i != _numActive; ++i) {
    const unsigned n = _activeIndices[i];
    // 1 2  4 8  16 32  64 128  256 512
    // 1 1
    //      1 1
    //           1  1   1
    _activeMasks[i] &= _mm_movemask_ps
                       (_mm_cmplt_ps(r2, _mm_add_ps(_mm_add_ps(_dx[n & 0x3],
                                     _dy[(n >> 2) & 0x3]),
                                     _dz[n >> 4])));
  }
  // Move the invalid points.
  std::size_t i = 0;
  while (i != _numActive) {
    // If each of the the points in the SIMD vector are inside the ball.
    if (_activeMasks[i] == 0) {
      // Move it to the inactive range.
      --_numActive;
      _activeIndices[i] = _activeIndices[_numActive];
      _activeMasks[i] = _activeMasks[_numActive];
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
}


// Return the number of grid points with negative distance.
inline
std::size_t
PatchCountNegative::
numNegative() const
{
  // The number of one bits in the lower four.
  const unsigned char CountBits[] = {
    0, 1, 1, 2,
    1, 2, 2, 3,
    1, 2, 2, 3,
    2, 3, 3, 4
  };

  // First count the active (non-negative) elements among the active SIMD
  // vectors.
  std::size_t count = 0;
  for (std::size_t i = 0; i != _numActive; ++i) {
    count += CountBits[_activeMasks[i]];
  }
  return NumPoints - count;
}


#endif

} // namespace levelSet
}

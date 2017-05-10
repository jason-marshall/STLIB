// -*- C++ -*-

#if !defined(stlib_simd_shuffle_tcc)
#error This file is an implementation detail of shuffle.
#endif

namespace stlib
{
namespace simd
{


// The generic implementation is not efficient because it does not utilize
// SIMD operations.
template<typename _Float, typename _D>
inline
void
_aosToSoa(_Float* p, _D /*Dimension*/)
{
  const std::size_t VectorSize = simd::Vector<_Float>::Size;
  const std::size_t D = _D::value;
  std::array<_Float, VectorSize* D> buffer;
  _Float* b = &buffer[0];
  for (std::size_t i = 0; i != D; ++i) {
    std::size_t k = i;
    for (std::size_t j = 0; j != VectorSize; ++j, k += D) {
      *b++ = p[k];
    }
  }
  memcpy(p, &buffer[0], VectorSize * D * sizeof(_Float));
}


// The generic implementation is not efficient because it does not utilize
// SIMD operations.
template<typename _Float, typename _D>
inline
void
_soaToAos(_Float* p, _D /*Dimension*/)
{
  const std::size_t VectorSize = simd::Vector<_Float>::Size;
  const std::size_t D = _D::value;
  std::array<_Float, VectorSize* D> buffer;
  _Float* b = &buffer[0];
  for (std::size_t i = 0; i != VectorSize; ++i) {
    std::size_t k = i;
    for (std::size_t j = 0; j != D; ++j, k += VectorSize) {
      *b++ = p[k];
    }
  }
  memcpy(p, &buffer[0], VectorSize * D * sizeof(_Float));
}


/// In 1-D, no conversion is necessary.
template<typename _Float>
inline
void
_aosToSoa(_Float* /*p*/, std::integral_constant<std::size_t, 1> /*Dimension*/)
{
}


/// In 1-D, no conversion is necessary.
template<typename _Float>
inline
void
_soaToAos(_Float* /*p*/, std::integral_constant<std::size_t, 1> /*Dimension*/)
{
}


//---------------------------------------------------------------------------
// AVX
//---------------------------------------------------------------------------
#ifdef __AVX__


/// Convert x0y0z0 ... x7y7z7 to x0...x7 y0...y7 z0...z7.
/**
  \pre The memory must be 32-byte alligned.

  Taken from "3D Vector Normalization Using 256-Bit Intel Advanced Vector
  Extensions (Intel AVX)" by Stan Melax.
*/
inline
void
_aosToSoa(float* p, std::integral_constant<std::size_t, 3> /*Dimension*/)
{
#ifdef STLIB_DEBUG
  // The memory must be 32-byte alligned.
  assert(std::size_t(p) % 32 == 0);
#endif
  __m128* m = (__m128*)p;
  // load lower halves
  __m256 m03 = _mm256_castps128_ps256(m[0]);
  __m256 m14 = _mm256_castps128_ps256(m[1]);
  __m256 m25 = _mm256_castps128_ps256(m[2]);
  // load upper halves
  m03 = _mm256_insertf128_ps(m03, m[3], 1);
  m14 = _mm256_insertf128_ps(m14, m[4], 1);
  m25 = _mm256_insertf128_ps(m25, m[5], 1);

  // upper x's and y's
  __m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2));
  // lower y's and z's
  __m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1));
  _mm256_store_ps(p, _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0)));
  _mm256_store_ps(p + 8, _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0)));
  _mm256_store_ps(p + 16, _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1)));
}


/// Convert x0...x7 y0...y7 z0...z7 to x0y0z0 ... x7y7z7.
/**
  \pre The memory must be 32-byte alligned.

  Taken from "3D Vector Normalization Using 256-Bit Intel Advanced Vector
  Extensions (Intel AVX)" by Stan Melax.
*/
inline
void
_soaToAos(float* p, std::integral_constant<std::size_t, 3> /*Dimension*/)
{
#ifdef STLIB_DEBUG
  // The memory must be 32-byte alligned.
  assert(std::size_t(p) % 32 == 0);
#endif

  // Starting SOA data
  __m256 x = _mm256_load_ps(p);
  __m256 y = _mm256_load_ps(p + 8);
  __m256 z = _mm256_load_ps(p + 16);

  __m256 rxy = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
  __m256 ryz = _mm256_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
  __m256 rzx = _mm256_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

  __m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2, 0, 2, 0));
  __m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3, 1, 2, 0));
  __m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3, 1, 3, 1));

  // _mm256_set_m128 may not be defined. We use insert instead.
  _mm256_store_ps(p, _mm256_insertf128_ps(r03, _mm256_castps256_ps128(r14), 1));
  _mm256_store_ps(p + 8,
                  _mm256_insertf128_ps(r03, _mm256_castps256_ps128(r25), 0));
  _mm256_store_ps(p + 16,
                  _mm256_insertf128_ps(r25, _mm256_extractf128_ps(r14, 1), 0));
}


//---------------------------------------------------------------------------
// SSE
//---------------------------------------------------------------------------
#elif defined(__SSE__)


/// Convert x0y0z0 x1y1z1 x2y2z2 x3y3z3 to x0x1x2x3 y0y1y2y3 z0z1z2z3.
/**
  \pre The memory must be 16-byte alligned.

  Taken from "3D Vector Normalization Using 256-Bit Intel Advanced Vector
  Extensions (Intel AVX)" by Stan Melax.
*/
inline
void
_aosToSoa(float* p, std::integral_constant<std::size_t, 3> /*Dimension*/)
{
#ifdef STLIB_DEBUG
  // The memory must be 16-byte alligned.
  assert(std::size_t(p) % 16 == 0);
#endif
  __m128 x0y0z0x1 = _mm_load_ps(p);
  __m128 y1z1x2y2 = _mm_load_ps(p + 4);
  __m128 z2x3y3z3 = _mm_load_ps(p + 8);
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
  // x0x1x2x3
  _mm_store_ps(p, _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)));
  // y0y1y2y3
  _mm_store_ps(p + 4, _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2,
                                     0)));
  // z0z1z2z3
  _mm_store_ps(p + 8, _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3,
                                     1)));
}


/// Convert x0x1x2x3 y0y1y2y3 z0z1z2z3 to x0y0z0 x1y1z1 x2y2z2 x3y3z3.
/**
  \pre The memory must be 16-byte alligned.

  Taken from "3D Vector Normalization Using 256-Bit Intel Advanced Vector
  Extensions (Intel AVX)" by Stan Melax.
*/
inline
void
_soaToAos(float* p, std::integral_constant<std::size_t, 3> /*Dimension*/)
{
#ifdef STLIB_DEBUG
  // The memory must be 16-byte alligned.
  assert(std::size_t(p) % 16 == 0);
#endif

  // Starting SOA data
  __m128 x = _mm_load_ps(p);
  __m128 y = _mm_load_ps(p + 4);
  __m128 z = _mm_load_ps(p + 8);

  __m128 x0x2y0y2 = _mm_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 y1y3z1z3 = _mm_shuffle_ps(y, z, _MM_SHUFFLE(3, 1, 3, 1));
  __m128 z0z2x1x3 = _mm_shuffle_ps(z, x, _MM_SHUFFLE(3, 1, 2, 0));

  _mm_store_ps(p, _mm_shuffle_ps(x0x2y0y2, z0z2x1x3, _MM_SHUFFLE(2, 0, 2, 0)));
  _mm_store_ps(p + 4, _mm_shuffle_ps(y1y3z1z3, x0x2y0y2, _MM_SHUFFLE(3, 1, 2,
                                     0)));
  _mm_store_ps(p + 8, _mm_shuffle_ps(z0z2x1x3, y1y3z1z3, _MM_SHUFFLE(3, 1, 3,
                                     1)));
}


#endif


/// Shuffle to transform an AOS block to an SOA block.
template<std::size_t _D, typename _Float>
inline
void
aosToSoa(_Float* p)
{
  _aosToSoa(p, std::integral_constant<std::size_t, _D>());
}


/// Shuffle to transform an SOA block to an AOS block.
template<std::size_t _D, typename _Float>
inline
void
soaToAos(_Float* p)
{
  _soaToAos(p, std::integral_constant<std::size_t, _D>());
}


template<std::size_t _D, typename _Float>
inline
void
aosToHybridSoa(std::vector<_Float, simd::allocator<_Float> >* data)
{
  const std::size_t VectorSize = Vector<_Float>::Size;
  const std::size_t BlockSize = _D * VectorSize;
  // The number of points must be a multiple of the SIMD vector size.
  assert(data->size() % BlockSize == 0);
  for (std::size_t i = 0; i != data->size(); i += BlockSize) {
    aosToSoa<_D>(&(*data)[i]);
  }
}


template<typename _Float, std::size_t _D>
inline
void
aosToHybridSoa(const std::vector<std::array<_Float, _D> >& input,
               std::vector<_Float, simd::allocator<_Float> >* shuffled)
{
  aosToHybridSoa<_D>(input.begin(), input.end(), shuffled);
}


template<std::size_t _D, typename _RandomAccessIterator, typename _Float>
inline
void
aosToHybridSoa(_RandomAccessIterator begin, _RandomAccessIterator const end,
               std::vector<_Float, simd::allocator<_Float> >* shuffled)
{
  typedef std::array<_Float, _D> Point;
  std::size_t const VectorSize = Vector<_Float>::Size;

  // Resize the output vector. Pad if necessary.
  std::size_t const size = std::distance(begin, end);
  shuffled->resize((size + VectorSize - 1) / VectorSize * VectorSize *
                   _D);
  // Copy the data.
  if (sizeof(Point) == _D * sizeof(_Float)) {
    memcpy(&(*shuffled)[0], &*begin, size * sizeof(Point));
  }
  else {
    std::size_t n = 0;
    for ( ; begin != end; ++begin) {
      for (std::size_t j = 0; j != _D; ++j, ++n) {
        (*shuffled)[n] = (*begin)[j];
      }
    }
  }
  // Pad with NaN's if necessary.
  memset(&(*shuffled)[0] + size * _D, -1,
         (shuffled->size() - size * _D) * sizeof(_Float));
  // Shuffle the coordinates.
  aosToHybridSoa<_D>(shuffled);
}


template<std::size_t _D, typename _Float>
inline
void
hybridSoaToAos(std::vector<_Float, simd::allocator<_Float> >* data)
{
  const std::size_t VectorSize = Vector<_Float>::Size;
  const std::size_t BlockSize = _D * VectorSize;
  // The number of points must be a multiple of the SIMD vector size.
  assert(data->size() % BlockSize == 0);
  for (std::size_t i = 0; i != data->size(); i += BlockSize) {
    soaToAos<_D>(&(*data)[i]);
  }
}


//--------------------------------------------------------------------------
// CONTINUE: REMOVE old implementations.

/// Convert x0y0z0 x1y1z1 x2y2z2 x3y3z3 to x0x1x2x3 y0y1y2y3 z0z1z2z3.
/**
  \pre The memory must be 16-byte alligned.

  Taken from "3D Vector Normalization Using 256-Bit Intel Advanced Vector
  Extensions (Intel AVX)" by Stan Melax
*/
inline
void
aos4x3ToSoa3x4(float* p)
{
#ifdef STLIB_DEBUG
  // The memory must be 16-byte alligned.
  assert(std::size_t(p) % 16 == 0);
#endif
  __m128 x0y0z0x1 = _mm_load_ps(p);
  __m128 y1z1x2y2 = _mm_load_ps(p + 4);
  __m128 z2x3y3z3 = _mm_load_ps(p + 8);
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
  // x0x1x2x3
  _mm_store_ps(p, _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)));
  // y0y1y2y3
  _mm_store_ps(p + 4, _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2,
                                     0)));
  // z0z1z2z3
  _mm_store_ps(p + 8, _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3,
                                     1)));
}

/// Use aos4x3ToSoa3x4() to shuffle each group of 12 floats.
/**
  \param p The pointer to the data.
  \param numStruct The number of structures. The size of the array is three
  times this size.

  \pre The memory must be 16-byte alligned.
  \pre The number of points (or structures) must be a multiple of four.
*/
inline
void
aos4x3ToSoa3x4(float* p, std::size_t numStruct)
{
#ifdef STLIB_DEBUG
  // The number of points must be a multiple of four.
  assert(numStruct % 4 == 0);
#endif
  for (std::size_t i = 0; i != numStruct; i += 4, p += 12) {
    aos4x3ToSoa3x4(p);
  }
}

/// Use aos4x3ToSoa3x4() to shuffle each group of 12 floats.
/**
  \param data The vector of data.

  \pre The number of points (or structures) must be a multiple of four.
*/
inline
void
aos4x3ToSoa3x4(std::vector<float, simd::allocator<float, 16> >* data)
{
#ifdef STLIB_DEBUG
  // The number of points must be a multiple of four.
  assert(data->size() % 12 == 0);
#endif
  aos4x3ToSoa3x4(&(*data)[0], data->size() / 3);
}

/// Resize the output, copy the data, and shuffle from AOS to SOA.
/**
  \pre The number of points must be a multiple of four.
*/
inline
void
aos4x3ToSoa3x4(const std::vector<std::array<float, 3> >& input,
               std::vector<float, simd::allocator<float, 16> >* shuffled)
{
  const std::size_t Dimension = 3;
  typedef std::array<float, Dimension> Point;

  // Resize the output vector.
  shuffled->resize(Dimension * input.size());
  // Copy the data.
  if (sizeof(Point) == Dimension * sizeof(float)) {
    memcpy(&(*shuffled)[0], &input[0], input.size() * sizeof(Point));
  }
  else {
    std::size_t n = 0;
    for (std::size_t i = 0; i != input.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j, ++n) {
        (*shuffled)[n] = input[i][j];
      }
    }
  }
  // Shuffle the coordinates.
  aos4x3ToSoa3x4(shuffled);
}


} // namespace simd
}

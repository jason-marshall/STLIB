// -*- C++ -*-

/**
  \file
  \brief Various bit manipulation utilities.
*/

#if !defined(__numerical_integer_bits_h__)
#define __numerical_integer_bits_h__

#include <boost/config.hpp>

#include <array>
#include <algorithm>
#include <vector>
#include <limits>

#include <cassert>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif

namespace stlib
{
namespace numerical
{

namespace
{

template<int enumeration>
struct
    IntegerTypes;

template<>
struct
    IntegerTypes<1> {
  typedef long Type;
};

template<>
struct
    IntegerTypes<2> {
  typedef int Type;
};

template<>
struct
    IntegerTypes<3> {
  typedef short Type;
};

template<>
struct
    IntegerTypes<4> {
  typedef char Type;
};

}

/** \defgroup numerical_integer_bits Bit Manipulation
@{
*/


/// Select an integer type with at least the specified number of bits.
/**
  The Boost library has a more sophisticated solution, but this is sufficient
  for my purposes.
*/
template<int Bits>
struct
    Integer {
  typedef typename IntegerTypes <
  (Bits - 1 <= std::numeric_limits<long>::digits) +
  (Bits - 1 <= std::numeric_limits<int>::digits) +
  (Bits - 1 <= std::numeric_limits<short>::digits) +
  (Bits - 1 <= std::numeric_limits<char>::digits) >::Type Type;
};



namespace
{

template<int enumeration>
struct
    UnsignedIntegerTypes;

template<>
struct
    UnsignedIntegerTypes<1> {
  typedef unsigned long Type;
};

template<>
struct
    UnsignedIntegerTypes<2> {
  typedef unsigned int Type;
};

template<>
struct
    UnsignedIntegerTypes<3> {
  typedef unsigned short Type;
};

template<>
struct
    UnsignedIntegerTypes<4> {
  typedef unsigned char Type;
};

}

/// Select an unsigned integer type with at least the specified number of bits.
/**
  The Boost library has a more sophisticated solution, but this is sufficient
  for my purposes.
*/
template<int Bits>
struct
    UnsignedInteger {
  typedef typename UnsignedIntegerTypes <
  (Bits <= std::numeric_limits<unsigned long>::digits) +
  (Bits <= std::numeric_limits<unsigned int>::digits) +
  (Bits <= std::numeric_limits<unsigned short>::digits) +
  (Bits <= std::numeric_limits<unsigned char>::digits) >::Type Type;
};

/// Fall-back if there are no intrinsics or built-ins for counting bits.
template<typename _Integer>
static inline
int
popCountByNibbles(_Integer x)
{
  // The number of one bits in a nibble (four bits).
  BOOST_STATIC_CONSTEXPR unsigned char CountBits[] = {
    0, 1, 1, 2,
    1, 2, 2, 3,
    1, 2, 2, 3,
    2, 3, 3, 4
  };

  int count = 0;
  const _Integer mask = 0xF;
  for (std::size_t i = 0; i != 2 * sizeof(_Integer); ++i) {
    count += CountBits[x & mask];
    x >>= 4;
  }
  return count;
}


/// Population count. Return the number of set bits.
inline
int
popCount(const unsigned x)
{
  // The GCC built-in __builtin_popcount() takes an unsigned int as argument.
  // In principle, this should be an option, but it triggers an
  // Illegal instruction/operand with GCC 4.6.
#ifdef __SSE4_2__
  static_assert(sizeof(unsigned) <= sizeof(std::uint32_t),
                "Invalid assumption.");
  return _mm_popcnt_u32(x);
#else
  return popCountByNibbles(x);
#endif
}

// Note that it won't work to define popCount() in terms of the std::uint*_t
// types because this may introduce ambiguities. For example, std::uint64_t
// might be defined as either unsigned long or unsigned long long. These two
// types are distinct even if they are the same size.

/// Population count. Return the number of set bits.
inline
int
popCount(unsigned long const x)
{
  // The GCC built-in __builtin_popcount() takes an unsigned int as argument.
  // In principle, this should be an option, but it triggers an
  // Illegal instruction/operand with GCC 4.6.
#ifdef __SSE4_2__
#ifdef __x86_64__
  static_assert(sizeof(unsigned long) <= sizeof(std::uint64_t),
                "Invalid assumption.");
  return _mm_popcnt_u64(x);
#else
  static_assert(sizeof(unsigned long) <= sizeof(std::uint32_t),
                "Invalid assumption.");
  return _mm_popcnt_u32(x);
#endif
#else
  return popCountByNibbles(x);
#endif
}


/// Population count. Return the number of set bits.
inline
int
popCount(unsigned long long const x)
{
#ifdef __SSE4_2__
  static_assert(sizeof(unsigned long long) <= sizeof(std::uint64_t),
                "Invalid assumption.");
  return _mm_popcnt_u64(x);
#else
  return popCountByNibbles(x);
#endif
}


/// Population count. Return the number of set bits.
inline
int
popCount(unsigned char const x)
{
  return popCount(unsigned(x));
}


/// Population count. Return the number of set bits.
inline
int
popCount(unsigned short const x)
{
  return popCount(unsigned(x));
}


/// Return the position of the highest set bit.
/** This is also floor(log_2(n)). The positions are zero-offset, starting
  at the least significant bit.

  \pre The argument must not be zero. */
template<typename _UnsignedInteger>
int
highestBitPosition(_UnsignedInteger n)
{
  assert(n != 0);
  int result = 0;
  while (n >>= 1) {
    ++result;
  }
  return result;
}

/// Extract bits, starting with the least significant.
template<typename _Integer, std::size_t _Size>
inline
void
getBits(_Integer n, std::array<bool, _Size>* bits)
{
  static_assert(_Size > 0, "Bad dimension.");
  (*bits)[0] = n % 2;
  for (std::size_t i = 1; i != _Size; ++i) {
    n /= 2;
    (*bits)[i] = n % 2;
  }
}

/// Extract bits, starting with the least significant.
template < typename _Integer, template<typename, std::size_t> class _Array,
           std::size_t _Size >
inline
void
getBits(_Integer n, _Array<bool, _Size>* bits)
{
  static_assert(_Size > 0, "Bad dimension.");
  (*bits)[0] = n % 2;
  for (std::size_t i = 1; i != _Size; ++i) {
    n /= 2;
    (*bits)[i] = n % 2;
  }
}

/// Reverse the bits of an unsigned integer type.
template<typename _Integer>
inline
_Integer
reverseBits(_Integer source)
{
  // Get the least significant bit.
  _Integer reversed = source;
  // The number of shifts we will make.
  int shift = std::numeric_limits<_Integer>::digits - 1;
  // Loop while there are non-zero bits left in the source.
  for (source >>= 1; source; source >>= 1) {
    reversed <<= 1;
    reversed |= source & 1;
    --shift;
  }
  // Do a shift when some of the source's most significant bits are zero.
  reversed <<= shift;

  return reversed;
}

/// Reverse the n least significant bits of an unsigned integer type.
/**
  The more significant bits will be zero.
*/
template<typename _Integer>
inline
_Integer
reverseBits(_Integer source, int n)
{
#ifdef STLIB_DEBUG
  assert(n >= 0);
#endif
  _Integer reversed = 0;
  for (; n; --n) {
    reversed <<= 1;
    reversed |= source & 1;
    source >>= 1;
  }
  return reversed;
}

/// Interlace the n least significant bits.
/**
  The return type must be specified explicitly.  It cannot be deduced from
  the arguments.
*/
template<typename _ResultInteger, std::size_t N, typename _ArgumentInteger>
inline
_ResultInteger
interlaceBits(std::array<_ArgumentInteger, N> sources,
              const std::size_t n)
{
  _ResultInteger reversed = 0;
  // Interlace in reverse order.
  for (std::size_t i = 0; i != n; ++i) {
    for (std::size_t j = 0; j != N; ++j) {
      reversed <<= 1;
      reversed |= sources[j] & 1;
      sources[j] >>= 1;
    }
  }
  // Reverse the bits.
  return reverseBits(reversed, n * N);
}

/// Interlace the n least significant bits.
/**
  The return type must be specified explicitly.  It cannot be deduced from
  the arguments.
*/
template < typename _ResultInteger, template<typename, int> class _Array,
           typename _ArgumentInteger, std::size_t N >
inline
_ResultInteger
interlaceBits(_Array<_ArgumentInteger, N> sources, const std::size_t n)
{
#ifdef STLIB_DEBUG
  assert(n >= 0);
#endif
  _ResultInteger reversed = 0;
  // Interlace in reverse order.
  for (std::size_t i = 0; i != n; ++i) {
    for (std::size_t j = 0; j != N; ++j) {
      reversed <<= 1;
      reversed |= sources[j] & 1;
      sources[j] >>= 1;
    }
  }
  // Reverse the bits.
  return reverseBits(reversed, n * N);
}

/// Unlace the n least significant bits in each coordinate.
/**
  The return type must be specified explicitly.  It cannot be deduced from
  the arguments.
*/
template<typename _SourceInteger, std::size_t N, typename _TargetInteger>
inline
void
unlaceBits(_SourceInteger source, const std::size_t n,
           std::array<_TargetInteger, N>* targets)
{
#ifdef STLIB_DEBUG
  assert(std::size_t(std::numeric_limits<_SourceInteger>::digits +
                     std::numeric_limits<_SourceInteger>::is_signed) >= n * N);
#endif

  // Clear the target coordinates.
  std::fill(targets->begin(), targets->end(), 0);
  // Unlace into reverse order targets.
  for (std::size_t i = 0; i != n; ++i) {
    for (std::size_t j = 0; j != N; ++j) {
      (*targets)[j] <<= 1;
      (*targets)[j] |= source & 1;
      source >>= 1;
    }
  }
  // Reverse the bits.
  for (std::size_t j = 0; j != N; ++j) {
    (*targets)[j] = reverseBits((*targets)[j], n);
  }
}

// CONTINUE
#if 0
/// Unlace the n least significant bits in each coordinate.
/**
  The return type must be specified explicitly.  It cannot be deduced from
  the arguments.
*/
template < typename _SourceInteger, template<typename, int> class _Array,
           typename _TargetInteger, std::size_t N >
inline
void
unlaceBits(_SourceInteger source, const std::size_t n,
           _Array<_TargetInteger, N>* targets)
{
#ifdef STLIB_DEBUG
  assert(n >= 0);
  assert(std::numeric_limits<_SourceInteger>::digits >= n * N);
#endif

  // Clear the target coordinates.
  *targets = _TargetInteger(0);
  // Unlace into reverse order targets.
  for (std::size_t i = 0; i != n; ++i) {
    for (std::size_t j = 0; j != N; ++j) {
      (*targets)[j] <<= 1;
      (*targets)[j] |= source & 1;
      source >>= 1;
    }
  }
  // Reverse the bits.
  for (std::size_t j = 0; j != N; ++j) {
    (*targets)[j] = reverseBits((*targets)[j], n);
  }
}
#endif


/// Convert a sequence of indices to a bit array.
/** \note You must explicitly specify the first template parameter. It cannot
 be deduced from the arguments. */
template<typename _UnsignedInteger, typename _Index>
inline
std::vector<_UnsignedInteger>
convertIndicesToBitArray(std::size_t const size,
                         std::vector<_Index> const& indices)
{
  // Determine the number of unsigned integers that we need for the bit array.
  // Initialize each bit to false.
  int const Digits = std::numeric_limits<_UnsignedInteger>::digits;
  std::vector<_UnsignedInteger>
    bitArray((size + Digits - 1) / Digits * Digits, 0);

  // Set the bits.
  for (auto i : indices) {
#ifdef STLIB_DEBUG
    assert(std::size_t(i) < size);
#endif
    bitArray[i / Digits] |= _UnsignedInteger(1) << (i % Digits);
  }
  return bitArray;
}


/// Convert a sequence of indices to a bit array.
/** \note You must explicitly specify the first template parameter. It cannot
 be deduced from the arguments. */
template<typename _Index, typename _UnsignedInteger>
inline
std::vector<_Index>
convertBitArrayToIndices(std::vector<_UnsignedInteger> const& bitArray)
{
  // Count the number of indices.
  std::size_t size = 0;
  for (auto bits : bitArray) {
    size += popCount(bits);
  }
  std::vector<_Index> indices;
  indices.reserve(size);

  // Record the indices.
  int const Digits = std::numeric_limits<_UnsignedInteger>::digits;
  for (std::size_t block = 0; block != bitArray.size(); ++block) {
    if (bitArray[block]) {
      _UnsignedInteger bits = bitArray[block];
      _Index const offset = block * Digits;
      for (std::size_t i = 0; i != Digits; ++i, bits >>= 1) {
        if (bits & _UnsignedInteger(1)) {
          indices.push_back(offset + i);
        }
      }
    }
  }
  return indices;
}


/// Convert a sequence of indices to a bit vector.
template<typename _Index>
inline
std::vector<bool>
convertIndicesToBitVector(std::size_t const size,
                          std::vector<_Index> const& indices)
{
  std::vector<bool> bits(size, false);
  for (auto i : indices) {
    bits[i] = true;
  }
  return bits;
}


/// Convert a sequence of indices to a bit array.
/** \note You must explicitly specify the template parameter. It cannot
  be deduced from the arguments. */
template<typename _Index>
inline
std::vector<_Index>
convertBitVectorToIndices(std::vector<bool> const& bits)
{
  std::vector<_Index> indices;
  indices.reserve(std::count(bits.begin(), bits.end(), true));
  for (std::size_t i = 0; i != bits.size(); ++i) {
    if (bits[i]) {
      indices.push_back(i);
    }
  }
  return indices;
}


/// @}

} // namespace numerical
} // namespace stlib

#endif

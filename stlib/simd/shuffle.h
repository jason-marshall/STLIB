// -*- C++ -*-

#ifndef stlib_simd_shuffle_h
#define stlib_simd_shuffle_h

#include "stlib/simd/allocator.h"
#include "stlib/simd/constants.h"


#include <array>
#include <vector>

#include <cassert>
#include <cstring>

namespace stlib
{
namespace simd
{


/// Shuffle to transform an AOS representation to a hybrid SOA representation.
/** The first template parameter, the number of elements in the structure,
 cannot be deduced from the function arguments; it must be specified
 explicitly. Let \c S be the default SIMD vector length. For example,
 with AVX, the SIMD vector length for single-precision floats is 8.
 The length of the array must be a multiple of \c _D*S. The use of
 simd::allocator guarantees that the data has the right alignment for
 SIMD operations.

 The input data is an array of structures. The structures are a sequence
 of floating-point values, perhaps a Cartesian point. The data is transposed
 to obtain a hybrid structure of arrays representation. It is \em hybrid
 because the transpose is only applied to blocks of length \c _D*S.
 This transformation is more storage-efficient than a global transpose because
 one only needs a buffer the size of a single block. The hybrid
 representation also has better locality properties than the plain
 SOA representation when one performs operations on the whole structure.
*/
template<std::size_t _D, typename _Float>
void
aosToHybridSoa(std::vector<_Float, simd::allocator<_Float> >* data);


/// Shuffle to transform an AOS representation to a hybrid SOA representation.
/** If necessary, the output will be resized. The size of the output vector
 will also be padded if necessary so that the number of points is a multiple
 of the SIMD vector length. */
template<typename _Float, std::size_t _D>
void
aosToHybridSoa(std::vector<std::array<_Float, _D> > const& input,
               std::vector<_Float, simd::allocator<_Float> >* shuffled);


/// Shuffle to transform an AOS representation to a hybrid SOA representation.
/** If necessary, the output will be resized. The size of the output vector
 will also be padded if necessary so that the number of points is a multiple
 of the SIMD vector length. */
template<std::size_t _D, typename _RandomAccessIterator, typename _Float>
void
aosToHybridSoa(_RandomAccessIterator begin, _RandomAccessIterator const end,
               std::vector<_Float, simd::allocator<_Float> >* shuffled);


/// Shuffle to transform a hybrid SOA representation to an AOS representation.
/** Reverse the transpose of
  aosToHybridSoa(std::vector<_Float, simd::allocator<_Float>>*). */
template<std::size_t _D, typename _Float>
void
hybridSoaToAos(std::vector<_Float, simd::allocator<_Float> >* data);


} // namespace simd
}

#define stlib_simd_shuffle_tcc
#include "stlib/simd/shuffle.tcc"
#undef stlib_simd_shuffle_tcc


#endif // stlib_simd_shuffle_h

// -*- C++ -*-

#if !defined(__sfc_HilbertOrder_tcc__)
#error This file is an implementation detail of HilbertOrder.
#endif

namespace stlib
{
namespace sfc
{


/* 
 * hilbert.c - Computes Hilbert space-filling curve coordinates, without
 * recursion, from integer index, and vice versa, and other Hilbert-related
 * calculations.  Also known as Pi-order or Peano scan.
 * 
 * Author:      Doug Moore
 *              Dept. of Computational and Applied Math
 *              Rice University
 *              http://www.caam.rice.edu/~dougm
 * Date:        Sun Feb 20 2000
 * Copyright (c) 1998-2000, Rice University
 *
 * Acknowledgement:
 * This implementation is based on the work of A. R. Butz ("Alternative
 * Algorithm for Hilbert's Space-Filling Curve", IEEE Trans. Comp., April,
 * 1971, pp 424-426) and its interpretation by Spencer W. Thomas, University
 * of Michigan (http://www-personal.umich.edu/~spencer/Home.html) in his widely
 * available C software.  While the implementation here differs considerably
 * from his, the first two interfaces and the style of some comments are very
 * much derived from his work. */

/* LICENSE
 *
 * This software is copyrighted by Rice University.  It may be freely copied,
 * modified, and redistributed, provided that the copyright notice is 
 * preserved on all copies.
 * 
 * There is no warranty or other guarantee of fitness for this software,
 * it is provided solely "as is".  Bug reports or fixes may be sent
 * to the author, who may or may not act on them as he desires.
 *
 * You may include this software in a program or other software product,
 * but must display the notice:
 *
 * Hilbert Curve implementation copyright 1998, Rice University
 *
 * in any place where the end-user would see your own copyright.
 * 
 * If you modify this software, you should include a notice giving the
 * name of the person performing the modification, the date of modification,
 * and the reason for such modification.
 */



/* Revision history:
   
   July 1998: Initial release

   Sept 1998: Second release

   Dec 1998: Fixed bug in hilbert_c2i that allowed a shift by number of bits in
   bitmask to vaporize index, in last bit of the function.  Implemented
   hilbert_incr.

   August 1999: Added argument to hilbert_nextinbox so that you can, optionally,
   find the previous point along the curve to intersect the box, rather than the
   next point.

   Nov 1999: Defined fast bit-transpose function (fast, at least, if the number
   of bits is large), and reimplemented i2c and c2i in terms of it.  Collapsed
   loops in hilbert_cmp, with the intention of reusing the cmp code to compare
   more general bitstreams.

   Feb 2000: Implemented almost all the floating point versions of cmp, etc, so
   that coordinates expressed in terms of double-precision IEEE floating point
   can be ordered.  Still have to do next-in-box, though.

   Oct 2001: Learned that some arbitrary coding choices caused some routines
   to fail in one dimension, and changed those choices.

   version 2001-10-20-05:34
   
*/

/* define the bitmask_t type as an integer of sufficient size */
typedef unsigned long long bitmask_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
typedef unsigned long halfmask_t;

#define adjust_rotation(rotation,nDims,bits)                            \
do {                                                                    \
      /* rotation = (rotation + 1 + ffs(bits)) % nDims; */              \
      bits &= -bits & nd1Ones;                                          \
      while (bits)                                                      \
        bits >>= 1, ++rotation;                                         \
      if ( ++rotation >= nDims )                                        \
        rotation -= nDims;                                              \
} while (0)

#define ones(T,k) ((((T)2) << (k-1)) - 1)

#define rdbit(w,k) (((w) >> (k)) & 1)
     
#define rotateRight(arg, nRots, nDims)                                  \
((((arg) >> (nRots)) | ((arg) << ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define rotateLeft(arg, nRots, nDims)                                   \
((((arg) << (nRots)) | ((arg) >> ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define DLOGB_BIT_TRANSPOSE
static bitmask_t
bitTranspose(unsigned nDims, unsigned nBits, bitmask_t inCoords)
#if defined(DLOGB_BIT_TRANSPOSE)
{
  unsigned const nDims1 = nDims-1;
  unsigned inB = nBits;
  unsigned utB;
  bitmask_t inFieldEnds = 1;
  bitmask_t inMask = ones(bitmask_t,inB);
  bitmask_t coords = 0;

  while ((utB = inB / 2))
    {
      unsigned const shiftAmt = nDims1 * utB;
      bitmask_t const utFieldEnds =
        inFieldEnds | (inFieldEnds << (shiftAmt+utB));
      bitmask_t const utMask =
        (utFieldEnds << utB) - utFieldEnds;
      bitmask_t utCoords = 0;
      unsigned d;
      if (inB & 1)
        {
          bitmask_t const inFieldStarts = inFieldEnds << (inB-1);
          unsigned oddShift = 2*shiftAmt;
          for (d = 0; d < nDims; ++d)
            {
              bitmask_t in = inCoords & inMask;
              inCoords >>= inB;
              coords |= (in & inFieldStarts) << oddShift++;
              in &= ~inFieldStarts;
              in = (in | (in << shiftAmt)) & utMask;
              utCoords |= in << (d*utB);
            }
        }
      else
        {
          for (d = 0; d < nDims; ++d)
            {
              bitmask_t in = inCoords & inMask;
              inCoords >>= inB;
              in = (in | (in << shiftAmt)) & utMask;
              utCoords |= in << (d*utB);
            }
        }
      inCoords = utCoords;
      inB = utB;
      inFieldEnds = utFieldEnds;
      inMask = utMask;
    }
  coords |= inCoords;
  return coords;
}
#else
{
  bitmask_t coords = 0;
  unsigned d;
  for (d = 0; d < nDims; ++d)
    {
      unsigned b;
      bitmask_t in = inCoords & ones(bitmask_t,nBits);
      bitmask_t out = 0;
      inCoords >>= nBits;
      for (b = nBits; b--;)
        {
          out <<= nDims;
          out |= rdbit(in, b);
        }
      coords |= out << d;
    }
  return coords;
}
#endif

/*****************************************************************
 * hilbert_i2c
 * 
 * Convert an index into a Hilbert curve to a set of coordinates.
 * Inputs:
 *  nDims:      Number of coordinate axes.
 *  nBits:      Number of bits per axis.
 *  index:      The index, contains nDims*nBits bits
 *              (so nDims*nBits must be <= 8*sizeof(bitmask_t)).
 * Outputs:
 *  coord:      The list of nDims coordinates, each with nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof index) * (bits_per_byte)
 */
void
hilbert_i2c(unsigned nDims, unsigned nBits, bitmask_t index, bitmask_t coord[])
{
  if (nDims > 1)
    {
      bitmask_t coords;
      halfmask_t const nbOnes = ones(halfmask_t,nBits);
      unsigned d;

      if (nBits > 1)
        {
          unsigned const nDimsBits = nDims*nBits;
          halfmask_t const ndOnes = ones(halfmask_t,nDims);
          halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
          unsigned b = nDimsBits;
          unsigned rotation = 0;
          halfmask_t flipBit = 0;
          bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes;
          index ^= (index ^ nthbits) >> 1;
          coords = 0;
          do
            {
              halfmask_t bits = (index >> (b-=nDims)) & ndOnes;
              coords <<= nDims;
              coords |= rotateLeft(bits, rotation, nDims) ^ flipBit;
              flipBit = (halfmask_t)1 << rotation;
              adjust_rotation(rotation,nDims,bits);
            } while (b);
          for (b = nDims; b < nDimsBits; b *= 2)
            coords ^= coords >> b;
          coords = bitTranspose(nBits, nDims, coords);
        }
      else
        coords = index ^ (index >> 1);

      for (d = 0; d < nDims; ++d)
        {
          coord[d] = coords & nbOnes;
          coords >>= nBits;
        }
    }
  else
    coord[0] = index;
}

/*****************************************************************
 * hilbert_c2i
 * 
 * Convert coordinates of a point on a Hilbert curve to its index.
 * Inputs:
 *  nDims:      Number of coordinates.
 *  nBits:      Number of bits/coordinate.
 *  coord:      Array of n nBits-bit coordinates.
 * Outputs:
 *  index:      Output index value.  nDims*nBits bits.
 * Assumptions:
 *      nDims*nBits <= (sizeof bitmask_t) * (bits_per_byte)
 */
bitmask_t
hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])
{
  if (nDims > 1)
    {
      unsigned const nDimsBits = nDims*nBits;
      bitmask_t index;
      unsigned d;
      bitmask_t coords = 0;
      for (d = nDims; d--; )
        {
          coords <<= nBits;
          coords |= coord[d];
        }

      if (nBits > 1)
        {
          halfmask_t const ndOnes = ones(halfmask_t,nDims);
          halfmask_t const nd1Ones= ndOnes >> 1; /* for adjust_rotation */
          unsigned b = nDimsBits;
          unsigned rotation = 0;
          halfmask_t flipBit = 0;
          bitmask_t const nthbits = ones(bitmask_t,nDimsBits) / ndOnes;
          coords = bitTranspose(nDims, nBits, coords);
          coords ^= coords >> nDims;
          index = 0;
          do
            {
              halfmask_t bits = (coords >> (b-=nDims)) & ndOnes;
              bits = rotateRight(flipBit ^ bits, rotation, nDims);
              index <<= nDims;
              index |= bits;
              flipBit = (halfmask_t)1 << rotation;
              adjust_rotation(rotation,nDims,bits);
            } while (b);
          index ^= nthbits >> 1;
        }
      else
        index = coords;
      for (d = 1; d < nDimsBits; d *= 2)
        index ^= index >> d;
      return index;
    }
  else
    return coord[0];
}


template<std::size_t _Dimension, typename _Code>
inline
_Code
HilbertOrder<_Dimension, _Code>::
code(std::array<_Code, _Dimension> const& indices,
     std::size_t const numLevels) const
{
  return hilbert_c2i(_Dimension, numLevels,
                     &ext::ConvertArray<bitmask_t>::convert(indices)[0]);
}


} // namespace sfc
} // namespace stlib

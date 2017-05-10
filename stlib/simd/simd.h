// -*- C++ -*-

#if !defined(stlib_simd_simd_h)
#define stlib_simd_simd_h

#include "stlib/simd/macros.h"

#if !defined(STLIB_NO_SIMD_INTRINSICS) && !defined(__AVX__) && !defined(__SSE__)
#error Neither AVX nor SSE are enabled.
#endif

#include "stlib/simd/align.h"
#include "stlib/simd/allocator.h"
#include "stlib/simd/array.h"
#include "stlib/simd/constants.h"
#include "stlib/simd/functions.h"
#include "stlib/simd/operators.h"
#include "stlib/simd/shuffle.h"

namespace stlib
{
/// Classes, functions, and constants that ease the task of SIMD programming.
namespace simd
{
}
}

/**
\mainpage SIMD Programming

\section simdIntroduction Introduction

\par
This package has classes, functions, and constants that ease the task of
SIMD programming. Specifically, it deals with intra-register vectorization
using SSE and AVX.
Visit the Intel Developer Zone for information on the 
<a href="https://software.intel.com/en-us/intel-isa-extensions">
Intel Instruction Set Architecture Extensions</a>.

\par Alignment.
align.h has macros to help with data alignment. Use the ALIGN_SIMD macro to 
give objects the proper alignment for the highest support instruction set.
For example if AVX is enabled then the following would yield an array with 
32-byte alignment.
\code
ALIGN_SIMD float a[8];
\endcode
You can use the stlib::simd::isAligned() function to check if an address
is properly aligned for the highest supported instruction set.
Continuing the above example, the following code would verify the 
alignment of the array.
\code
assert(stlib::simd::isAligned(a));
\endcode

*/

/**
\page simdBibliography Bibliography

-# \anchor simdBoostSimd
<a href="http://www.numscale.com/boost-simd/">Boost.SIMD</a>

*/

#endif

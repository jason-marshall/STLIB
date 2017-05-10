// -*- C++ -*-

#if !defined(stlib_simd_macros_h)
#define stlib_simd_macros_h

// MSVC does not define the __SSE*__ macros. For 64-bit architectures, either
// _M_AMD64 or _M_X64 are defined. SSE2 instructions are enabled in 
// this case. For 32-bit architectures:
// _M_IX86_FP == 0 means SSE instructions are disabled,
// _M_IX86_FP == 1 means SSE instructions are enabled,
// _M_IX86_FP == 2 means SSE2 instructions are enabled.
#ifdef _MSC_VER

// Note that someone else may have already done what I'm doing here, so I 
// first check that __SSE__ has not already been defined.
#ifndef __SSE__
#if (defined(_M_AMD64) || defined(_M_X64) || _M_IX86_FP >= 1)
#define __SSE__
#endif
#endif

#ifndef __SSE2__
#if (defined(_M_AMD64) || defined(_M_X64) || _M_IX86_FP == 2)
#define __SSE2__
#endif
#endif

#endif

#endif

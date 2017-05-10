// -*- C++ -*-

// GCC 4.0-4.2 supports up to SSE 3
// GCC 4.3 supports up to SSE 5
// GCC 4.4 supports up to AVX

// Apple GCC 4.2 supports up to SSE 4

// SSE     xmmintrin.h
// SSE 2   emmintrin.h
// SSE 3   pmmintrin.h
// SSSE 3  tmmintrin.h
// SSE 4a  ammintrin.h
// SSE 4.1 smmintrin.h
// SSE 4.2 nmmintrin.h
// AVX     immintrin.h

#include "stlib/simd/macros.h"

#include <iostream>

int
main()
{
  std::cout << "The following technologies are enabled:\n"
#ifdef __SSE__
            << "SSE\n"
#endif
#ifdef __SSE2__
            << "SSE 2\n"
#endif
#ifdef __SSE3__
            << "SSE 3\n"
#endif
#ifdef __SSSE3__
            << "SSSE 3\n"
#endif
#ifdef __SSE4A__
            << "SSE 4A\n"
#endif
#ifdef __SSE4_1__
            << "SSE 4.1\n"
#endif
#ifdef __SSE4_2__
            << "SSE 4.2\n"
#endif
#ifdef __AVX__
            << "AVX\n"
#endif
#ifdef __AVX2__
            << "AVX2\n"
#endif
#ifdef __AVX512F__
            << "AVX-512F\n"
#endif
            ;

  return 0;
}

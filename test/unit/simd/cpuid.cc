// -*- C++ -*-

#include "stlib/simd/cpuid.h"

#include <iostream>

using namespace stlib;

int
main()
{
  std::cout << "SSE2: " << simd::hasSSE2() << '\n'
            << "SSE41: " << simd::hasSSE41() << '\n'
            << "SSE42: " << simd::hasSSE42() << '\n'
            << "AVX: " << simd::hasAVX() << '\n'
            << "AVX2: " << simd::hasAVX2() << '\n'
            << "AVX512F: " << simd::hasAVX512F() << '\n';

  return 0;
}

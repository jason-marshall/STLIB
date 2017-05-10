// -*- C++ -*-

#include "stlib/simd/cpuid.h"

using namespace stlib;

void
print_AVX();

void
print_SSE2();

void
print()
{
  if (simd::hasAVX()) {
    print_AVX();
  }
  else if (simd::hasSSE2()) {
    print_SSE2();
  }
  else {
    throw std::runtime_error("No suitable SIMD instruction set.");
  }
}

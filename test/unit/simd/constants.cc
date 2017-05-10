// -*- C++ -*-

#include "stlib/simd/constants.h"

using namespace stlib;

int
main()
{
  // Default register size.
#ifdef STLIB_AVX512F
  static_assert(simd::Vector<float>::Size == 16, "Error.");
  static_assert(simd::Vector<double>::Size == 8, "Error.");
  static_assert(simd::Vector<unsigned>::Size == 16, "Error.");
#elif defined(__AVX__)
  static_assert(simd::Vector<float>::Size == 8, "Error.");
  static_assert(simd::Vector<double>::Size == 4, "Error.");
#elif defined(__SSE__)
  static_assert(simd::Vector<float>::Size == 4, "Error.");
  static_assert(simd::Vector<double>::Size == 2, "Error.");
#else
  static_assert(simd::Vector<float>::Size == 1, "Error.");
  static_assert(simd::Vector<double>::Size == 1, "Error.");
#endif

#ifdef STLIB_AVX512F
  static_assert(simd::Vector<unsigned>::Size == 16, "Error.");
#elif defined(__AVX2__)
  static_assert(simd::Vector<unsigned>::Size == 8, "Error.");
#elif defined(__SSE2__)
  static_assert(simd::Vector<unsigned>::Size == 4, "Error.");
#else
  static_assert(simd::Vector<unsigned>::Size == 1, "Error.");
#endif

  static_assert(sizeof(simd::Vector<float>::Type) ==
                    (simd::Vector<float>::Size) * sizeof(float), "Error.");
  static_assert(sizeof(simd::Vector<double>::Type) ==
                    (simd::Vector<double>::Size) * sizeof(double), "Error.");
  static_assert(sizeof(simd::Vector<unsigned>::Type) ==
                    (simd::Vector<unsigned>::Size) * sizeof(unsigned), "Error.");

  // Specified register size.
#ifdef STLIB_AVX512F
  static_assert((simd::Vector<float, 16>::Size) == 16, "Error.");
  static_assert((simd::Vector<double, 8>::Size) == 8, "Error.");
  static_assert((simd::Vector<unsigned, 16>::Size) == 16, "Error.");
  static_assert(sizeof(simd::Vector<float, 16>::Type) ==
                    (simd::Vector<float, 16>::Size) * sizeof(float), "Error.");
  static_assert(sizeof(simd::Vector<double, 8>::Type) ==
                    (simd::Vector<double, 8>::Size) * sizeof(double), "Error.");
  static_assert(sizeof(simd::Vector<unsigned, 16>::Type) ==
                    (simd::Vector<unsigned, 16>::Size) * sizeof(unsigned),
                    "Error.");
#endif
#ifdef __AVX2__
  static_assert((simd::Vector<unsigned, 8>::Size) == 8, "Error.");
  static_assert(sizeof(simd::Vector<unsigned, 8>::Type) ==
                    (simd::Vector<unsigned, 8>::Size) * sizeof(unsigned),
                    "Error.");
#endif
#ifdef __AVX__
  static_assert((simd::Vector<float, 8>::Size) == 8, "Error.");
  static_assert((simd::Vector<double, 4>::Size) == 4, "Error.");
  static_assert(sizeof(simd::Vector<float, 8>::Type) ==
                    (simd::Vector<float, 8>::Size) * sizeof(float), "Error.");
  static_assert(sizeof(simd::Vector<double, 4>::Type) ==
                    (simd::Vector<double, 4>::Size) * sizeof(double), "Error.");
#endif
#ifdef __SSE2__
  static_assert((simd::Vector<unsigned, 4>::Size) == 4, "Error.");
  static_assert(sizeof(simd::Vector<unsigned, 4>::Type) ==
                    (simd::Vector<unsigned, 4>::Size) * sizeof(unsigned),
                    "Error.");
#endif
#ifdef __SSE__
  static_assert((simd::Vector<float, 4>::Size) == 4, "Error.");
  static_assert((simd::Vector<double, 2>::Size) == 2, "Error.");
  static_assert(sizeof(simd::Vector<float, 4>::Type) ==
                    (simd::Vector<float, 4>::Size) * sizeof(float), "Error.");
  static_assert(sizeof(simd::Vector<double, 2>::Type) ==
                    (simd::Vector<double, 2>::Size) * sizeof(double), "Error.");
#endif
  static_assert((simd::Vector<float, 1>::Size) == 1, "Error.");
  static_assert((simd::Vector<double, 1>::Size) == 1, "Error.");
  static_assert((simd::Vector<unsigned, 1>::Size) == 1, "Error.");
  static_assert(sizeof(simd::Vector<float, 1>::Type) ==
                    (simd::Vector<float, 1>::Size) * sizeof(float), "Error.");
  static_assert(sizeof(simd::Vector<double, 1>::Type) ==
                    (simd::Vector<double, 1>::Size) * sizeof(double), "Error.");
  static_assert(sizeof(simd::Vector<unsigned, 1>::Type) ==
                    (simd::Vector<unsigned, 1>::Size) * sizeof(unsigned),
                    "Error.");


  static_assert(sizeof(float) == 4, "Error.");
  {
    typedef float T;
    typedef simd::Vector<T> V;
    static_assert(sizeof(V::Type) == V::Size * sizeof(T), "Error.");
  }
  {
    typedef double T;
    typedef simd::Vector<T> V;
    static_assert(sizeof(V::Type) == V::Size * sizeof(T), "Error.");
  }

  static_assert(sizeof(unsigned) == 4, "Error.");
  {
    typedef unsigned T;
    typedef simd::Vector<T> V;
    static_assert(sizeof(V::Type) == V::Size * sizeof(T), "Error.");
  }

  return 0;
}

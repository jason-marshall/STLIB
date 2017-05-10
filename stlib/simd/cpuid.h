// -*- C++ -*-

#ifndef stlib_simd_cpuid_h
#define stlib_simd_cpuid_h

#include <stdexcept>
#include <sstream>

#if _WIN32 || _WIN64
#include <intrin.h>
#endif
#if __GNUC__ || __clang__
#include <cpuid.h>
#endif


namespace stlib
{
namespace simd
{


#if _WIN32 || _WIN64
inline
void
getCpuid(unsigned int const level, unsigned int* eax, unsigned int* ebx,
         unsigned int* ecx, unsigned int* edx)
{
  int cpuInfo[4];
  __cpuid(cpuInfo, int(level));
  *eax = cpuInfo[0];
  *ebx = cpuInfo[1];
  *ecx = cpuInfo[2];
  *edx = cpuInfo[3];
}
#elif __GNUC__ || __clang__
inline
void
getCpuid(unsigned int const level, unsigned int* eax, unsigned int* ebx,
         unsigned int* ecx, unsigned int* edx)
{
  if (__get_cpuid(level, eax, ebx, ecx, edx) == 0) {
    std::ostringstream message;
    message << "The cpuid level " << level << " is not supported.";
    throw std::runtime_error(message.str());
  }
}
#else
inline
void
getCpuid(unsigned int const level, unsigned int* eax, unsigned int* ebx,
         unsigned int* ecx, unsigned int* edx)
{
   __asm__ __volatile__
     ("cpuid": "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) :
      "a" (level), "c" (0));
}
#endif


/// Return true if SSE 2 is supported.
inline
bool
hasSSE2()
{
  unsigned int a, b, c, d;
  getCpuid(0x1, &a, &b, &c, &d);
  return (1 << 26) & d;
}


/// Return true if SSE 4.1 is supported.
inline
bool
hasSSE41()
{
  unsigned int a, b, c, d;
  getCpuid(0x1, &a, &b, &c, &d);
  return (1 << 19) & c;
}


/// Return true if SSE 4.2 is supported.
inline
bool
hasSSE42()
{
  unsigned int a, b, c, d;
  getCpuid(0x1, &a, &b, &c, &d);
  return (1 << 20) & c;
}


/// Return true if AVX is supported.
inline
bool
hasAVX()
{
  unsigned int a, b, c, d;
  getCpuid(0x1, &a, &b, &c, &d);
  // CONTINUE: Check XGETBV.
  // ECX.OSXSAVE and ECX.AVX
  return (1 << 27) & c && (1 << 28) & c;
}


/// Return true if AVX2 is supported.
inline
bool
hasAVX2()
{
  unsigned int a, b, c, d;
  getCpuid(0, &a, &b, &c, &d);
  if (a < 7) {
    return false;
  }
  getCpuid(7, &a, &b, &c, &d);
  return (1 << 5) & b;
}


/// Return true if AVX512F is supported.
inline
bool
hasAVX512F()
{
  unsigned int a, b, c, d;
  getCpuid(0, &a, &b, &c, &d);
  if (a < 7) {
    return false;
  }
  getCpuid(7, &a, &b, &c, &d);
  return (1 << 16) & b;
}


} // namespace simd
}

#endif // stlib_simd_cpuid_h

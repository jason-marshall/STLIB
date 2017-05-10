// Return 0 if AVX is supported. Otherwise return -1.
#define cpuid(func,ax,bx,cx,dx)\
   __asm__ __volatile__ ("cpuid":                               \
                         "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));

int
main() {
   unsigned long a, b, c, d;
   cpuid(0x1, a, b, c, d);
   // CONTINUE: Check XGETBV.
   // ECX.OSXSAVE and ECX.AVX
   if (((1 << 27) & c) && ((1 << 28) & c)) {
      return 0;
   }
   return -1;
}

#if 0
#include <intrin.h>
 
int main()
{
    bool avxSupported = false;
 
    // Checking for AVX requires 3 things:
    // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
    //     instructions (allowing saving YMM registers on context
    //     switch)
    // 2) CPUID indicates support for AVX
    // 3) XGETBV indicates the AVX registers will be saved and
    //     restored on context switch
    //
    // Note that XGETBV is only available on 686 or later CPUs, so
    // the instruction needs to be conditionally run.
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
 
    bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
    bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;
 
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        // Check if the OS will save the YMM registers
        unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0×6) == 6;
    }
 
    if (avxSupported) {
       return 0;
    }
    else {
       return -1;
    }
}
#endif

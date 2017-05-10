// Return 0 if SSE 4.1 is supported. Otherwise return -1.
#define cpuid(func,ax,bx,cx,dx)\
   __asm__ __volatile__ ("cpuid":                               \
                         "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));

int
main() {
   unsigned long a, b, c, d;
   cpuid(0x1, a, b, c, d);
   // SSE4.1 is bit 19.
   if ((1 << 19) & c) {
      return 0;
   }
   return -1;
}

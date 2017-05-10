// -*- C++ -*-

#ifndef __performance_C_mixedInteger_shiftRight_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"

#include <vector>
#include <iostream>

using namespace stlib;

int
main() {
   ads::Timer timer;
   std::vector<T1> a(100, 1);
   std::vector<T2> b(100, 1);
   const std::size_t Count = 1000000;

   // Warm up.
   for (std::size_t n = 0; n != 1000; ++n) {
      for (std::size_t i = 0; i != a.size(); ++i) {
         a[i] >>= b[i];
      }
   }

   timer.tic();
   for (std::size_t n = 0; n != Count; ++n) {
      for (std::size_t i = 0; i != a.size(); ++i) {
         a[i] >>= b[i];
      }
   }
   double elapsedTime = timer.toc();

   std::cout
         << "Elapsed time = " << elapsedTime << '\n'
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Time per " << Name << " = " << elapsedTime / (a.size() * Count) * 1e9
         << " nanoseconds.\n"
         << "Meaningless result = " << a[0] << "\n";

   return 0;
}

// -*- C++ -*-

#ifndef __performance_C_math_f_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"

#include <iostream>

#include <cmath>

int
main() {
   Result result = 0;
   ads::Timer timer;
   const std::size_t Count = 100000000;
   const Number Epsilon = std::numeric_limits<Number>::epsilon();

   Number x = InitialValue;
   // Warm up.
   for (std::size_t n = 0; n != 1000; ++n) {
      result += FUNCTION(x);
      x += Epsilon;
   }

   timer.tic();
   for (std::size_t n = 0; n != Count; ++n) {
      result += FUNCTION(x);
      x += Epsilon;
   }
   double elapsedTime = timer.toc();

   std::cout
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Time per " << FunctionName << " = " << elapsedTime / Count * 1e9
         << " nanoseconds.\n"
         << "Meaningless result = " << result << "\n";

   return 0;
}

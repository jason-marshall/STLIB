// -*- C++ -*-

#ifndef __performance_numerical_random_exponential_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"

#include <iostream>

using namespace stlib;

int
main() {
   double result = 0;
   ExponentialGenerator::DiscreteUniformGenerator uniform;
   ExponentialGenerator random(&uniform);
   ads::Timer timer;
   const int Count = 100000000;

   // Warm up.
   for (int n = 0; n != 1000; ++n) {
      result = random();
   }

   timer.tic();
   for (int n = 0; n != Count; ++n) {
      result = random();
   }
   double elapsedTime = timer.toc();

   std::cout
         << "Meaningless result = " << result << "\n"
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Generated " << Count << " exponential random numbers in "
         << elapsedTime << " seconds.\n"
         << "Time per random number = " << elapsedTime / Count * 1e9
         << " nanoseconds.\n";

   return 0;
}

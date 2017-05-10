// -*- C++ -*-

#ifndef __performance_C_math_branch_f_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/array/Array.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <algorithm>

int
main() {
   double result = 0;
   ads::Timer timer;
   const int Count = 1000000;

   ads::Array<1> x(100);
   for (int i = 0; i != x.size(); ++i) {
      x[i] = double(i) / (x.size() - 1);
   }
   std::random_shuffle(x.begin(), x.end());

   // Warm up.
   for (int n = 0; n != 1000; ++n) {
      for (int i = 0; i != x.size(); ++i) {
         result += function(x[i]);
      }
   }

   timer.tic();
   for (int n = 0; n != Count; ++n) {
      for (int i = 0; i != x.size(); ++i) {
         result += function(x[i]);
      }
   }
   double elapsedTime = timer.toc();

   std::cout
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Time per function call = "
         << elapsedTime / (Count * x.size()) * 1e9
         << " nanoseconds.\n"
         << "Meaningless result = " << result << "\n";

   return 0;
}

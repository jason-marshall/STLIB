// -*- C++ -*-

#ifndef __timeFunctor_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/array/Array.h"
#include <algorithm>

template<typename Generator>
inline
double
timeFunctor(Generator* random, const ads::Array<1, double>& means,
            const double targetTimePerTest) {
   ads::Timer timer;

   //
   // Determine an appropriate number of times to evaluate the functor.
   // Test the functor.
   //
   int count = 1;
   double time;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         for (int m = 0; m != means.size(); ++m) {
            (*random)(means[m]);
         }
      }
      time = timer.toc();
   }
   while (time < targetTimePerTest);
   // Return the time per call in nanoseconds.
   return timer.toc() / (count * means.size()) * 1e9;
}

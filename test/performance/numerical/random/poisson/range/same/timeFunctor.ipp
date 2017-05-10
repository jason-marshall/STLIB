// -*- C++ -*-

#ifndef __timeFunctor_ipp__
#error This file is an implementation detail.
#endif

template<typename Generator>
inline
double
timeFunctor(Generator* random, const double mean,
            const double targetTimePerTest) {
   ads::Timer timer;

   //
   // Determine an appropriate number of times to evaluate the functor.
   // Time the functor.
   //
   int count = 1000;
   double time;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         (*random)(mean);
      }
      time = timer.toc();
   }
   while (time < targetTimePerTest);
   // Return the time per call in nanoseconds.
   return timer.toc() / count * 1e9;
}

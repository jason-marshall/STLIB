// -*- C++ -*-

#ifndef __timeFunctor_ipp__
#error This file is an implementation detail.
#endif

template<typename Generator>
inline
double
timeFunctor(Generator* random, double mean,
            const double targetTimePerTest) {
   // A number, which when added to the mean, will change it by a small amount.
   // Use float because some methods use single precision.
   const double Epsilon = 2 * mean * std::numeric_limits<float>::epsilon();
   ads::Timer timer;

   //
   // Determine an appropriate number of times to evaluate the functor.
   //
   int count = 1000;
   double time;
   const double mean1 = mean;
   // Change the mean by a small amount.
   const double mean2 = mean + Epsilon;
   do {
      count *= 2;
      timer.tic();
      for (int n = 0; n != count; ++n) {
         (*random)(mean1);
         (*random)(mean2);
      }
      time = timer.toc();
   }
   while (time < targetTimePerTest);
   // Return the time per call in nanoseconds.
   return timer.toc() / (2 * count) * 1e9;
}

// -*- C++ -*-

#ifndef __test_numerical_random_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>


int
main() {
#if defined(POISSON_SIZE_CONSTRUCTOR)
   // No default constructor, copy constructor, or assignment operator.

   // Size constructor.
   CONSTRUCT(f, 101);
   {
      // Seed.
      CONSTRUCT(g, 101);
      g.seed(1);
   }
#elif defined(POISSON_THRESHHOLD_CONSTRUCTOR)
   // No default constructor, copy constructor, or assignment operator.

   // Size constructor.
   CONSTRUCT(f, 1e3);
   assert(f(1.0) >= 0);
   {
      // Seed.
      CONSTRUCT(g, 1e3);
      g.seed(1);
   }
#elif defined(POISSON_DOUBLE_THRESHHOLD_CONSTRUCTOR)
   // Size constructor.
   CONSTRUCT(f, 1e3, 1e6);
   assert(f(1.0) >= 0);
   {
      // Seed.
      CONSTRUCT(g, 1e3, 1e6);
      g.seed(1);
   }
#elif defined(POISSON_MAX_MEAN_CONSTRUCTOR)
   // Maximum mean constructor.
   CONSTRUCT(f, 32);
   {
      // Seed.
      CONSTRUCT(g, 32);
      g.seed(1);
   }
#else
   // Default constructor.
   CONSTRUCT(f);

   {
      // Copy constructor.
      PoissonGenerator g(f);
   }
   {
      // Assignment operator.
      CONSTRUCT(g);
      g = f;
   }
   {
      // Seed.
      CONSTRUCT(g);
      g.seed(1);
   }
#endif

   // Check the mean and variance.
   // CONTINUE: Numerically check the values instead of printing them.
   const int Size = 100000;
   std::vector<double> data(Size);
   const int NumberOfArguments = sizeof(Arguments) / sizeof(double);
   for (int i = 0; i != NumberOfArguments; ++i) {
      const double argument = Arguments[i];
#if defined(POISSON_SIZE_CONSTRUCTOR)
      const int maximumMean = int(argument);
      CONSTRUCT(p, maximumMean);
#elif defined(POISSON_THRESHHOLD_CONSTRUCTOR)
      CONSTRUCT(p, 1e3);
#elif defined(POISSON_DOUBLE_THRESHHOLD_CONSTRUCTOR)
      CONSTRUCT(p, 1e3, 1e6);
#elif defined(POISSON_MAX_MEAN_CONSTRUCTOR)
      CONSTRUCT(p, argument);
#else
      CONSTRUCT(p);
#endif
      for (int n = 0; n != Size; ++n) {
         data[n] = p(argument);
      }
      double mean = 0, variance = 0;
      ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
      std::cout << "Argument = " << argument << "\n"
                << "  Mean: actual = " << mean
                << ", expected = " << argument
                << ", difference = " << mean - argument << "\n"
                << "  Variance: actual = " << variance
                << ", expected = " << argument
                << ", difference = " << variance - argument << "\n";
      if (variance != 0) {
         double absoluteDeviation = 0, skew = 0, curtosis = 0;
         ads::computeMeanAbsoluteDeviationVarianceSkewAndCurtosis
         (data.begin(), data.end(), &mean, &absoluteDeviation, &variance,
          &skew, &curtosis);
         std::cout << "  Absolute deviation = " << absoluteDeviation << "\n"
                   << "  Skew: actual = " << skew
                   << ", expected = " << 1 / std::sqrt(argument)
                   << ", difference = " << skew - 1 / std::sqrt(argument) << "\n"
                   << "  Curtosis: actual = " << curtosis
                   << ", expected = " << 1 / argument
                   << ", difference = " << curtosis - 1 / argument << "\n";
      }
   }

   return 0;
}

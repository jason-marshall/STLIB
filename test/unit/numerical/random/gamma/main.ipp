// -*- C++ -*-

#ifndef __test_numerical_random_gamma_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>


int
main() {
   // Default constructor.
   GammaGenerator::DiscreteUniformGenerator uniform;
   GammaGenerator::NormalGenerator normal(&uniform);
   GammaGenerator f(&normal);
   assert(f(1.0) >= 0);

   {
      // Copy constructor.
      GammaGenerator g(f);
   }
   {
      // Assignment operator.
      GammaGenerator g(&normal);
      g = f;
   }
   {
      // Seed.
      GammaGenerator g(&normal);
      g.seed(1);
   }

   // Check the mean and variance.
   // CONTINUE: Numerically check the values instead of printing them.
   const int Size = 1000000;
   std::vector<double> data(Size);
   const int NumberOfArguments = sizeof(Arguments) / sizeof(double);
   for (int i = 0; i != NumberOfArguments; ++i) {
      const double argument = Arguments[i];
      for (int n = 0; n != Size; ++n) {
         data[n] = f(argument);
      }
      double mean = 0, absoluteDeviation = 0, variance = 0, skew = 0,
             curtosis = 0;
      ads::computeMeanAbsoluteDeviationVarianceSkewAndCurtosis
      (data.begin(), data.end(), &mean, &absoluteDeviation, &variance,
       &skew, &curtosis);
      std::cout << "Argument = " << argument << "\n"
                << "  Mean: actual = " << mean
                << ", expected = " << argument
                << ", difference = " << mean - argument << "\n";
      if (argument != 0) {
         std::cout << "  Absolute deviation = " << absoluteDeviation << "\n"
                   << "  Variance: actual = " << variance
                   << ", expected = " << argument
                   << ", difference = " << variance - argument << "\n"
                   << "  Skew: actual = " << skew
                   << ", expected = " << 2 / std::sqrt(argument)
                   << ", difference = " << skew - 2 / std::sqrt(argument) << "\n"
                   << "  Curtosis: actual = " << curtosis
                   << ", expected = " << 6 / argument
                   << ", difference = " << curtosis - 6 / argument << "\n";
      }
   }

   return 0;
}

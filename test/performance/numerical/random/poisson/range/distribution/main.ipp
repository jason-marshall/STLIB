// -*- C++ -*-

#ifndef __main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>


using namespace stlib;

#define __timeFunctor_ipp__
#include "timeFunctor.ipp"
#undef __timeFunctor_ipp__


int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the upper bound means.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the means file.\n";
      exit(1);
   }
   ads::Array<1, double> upperBoundMeans;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> upperBoundMeans;
   }

   // There should be no more arguments.
   if (! parser.areArgumentsEmpty()) {
      std::cerr << "Error: Un-parsed arguments:\n";
      parser.printArguments(std::cerr);
      exit(1);
   }

   // Get the target time per test.  The default is 1 second.
   double timePerTest = 1;
   parser.getOption("time", &timePerTest) ||
   parser.getOption("t", &timePerTest);
   if (timePerTest <= 0) {
      std::cerr << "Error: Bad target time per test.\n";
      exit(1);
   }

   // Get the number of distinct means.
   int numberOfMeans = 1;
   parser.getOption("number", &numberOfMeans) ||
   parser.getOption("n", &numberOfMeans);
   if (numberOfMeans < 1) {
      std::cerr << "Error: Bad number of distinct means.\n";
      exit(1);
   }

   // Get the multiplicity of each mean.
   int multiplicity = 1;
   parser.getOption("multiplicity", &multiplicity) ||
   parser.getOption("m", &multiplicity);
   if (multiplicity < 1) {
      std::cerr << "Error: Bad multiplicity.\n";
      exit(1);
   }

   // Get the lower bound for the distribution of means.
   // This is useful when the method has a problem with a mean of zero.
   double lowerBound = 0;
   parser.getOption("lower", &lowerBound) || parser.getOption("l", &lowerBound);
   if (lowerBound < 0) {
      std::cerr << "Error: Bad lower bound.\n";
      exit(1);
   }

#ifdef POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR
   // Get the normal threshhold.
   double normalThreshhold = 1000;
   parser.getOption("threshhold", &normalThreshhold);
   if (normalThreshhold < 0) {
      std::cerr << "Error: Bad normal threshhold.\n";
      exit(1);
   }
#endif

   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options.\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   ads::Array<1, double> time(upperBoundMeans.size());
   double meaninglessResult = 0;

   for (int n = 0; n != upperBoundMeans.size(); ++n) {
      // Build the uniform distribution of means.
      const double upperBoundMean = upperBoundMeans[n];
      ads::Array<1, double> means(numberOfMeans * multiplicity);
      int index = 0;
      int denominator = (numberOfMeans == 1 ? 1 : numberOfMeans - 1);
      for (int i = 0; i != numberOfMeans; ++i) {
         const double mean = lowerBound + i * (upperBoundMean - lowerBound) /
                             denominator;
         for (int j = 0; j != multiplicity; ++j) {
            means[index++] = mean;
         }
      }
      std::random_shuffle(means.begin(), means.end());

#if defined(POISSON_SIZE_CONSTRUCTOR)
      CONSTRUCT(random, int(upperBoundMean) + 1);
#elif defined(POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR)
      CONSTRUCT(random, normalThreshhold);
#elif defined(POISSON_MAX_MEAN_CONSTRUCTOR)
      CONSTRUCT(random, upperBoundMean);
#else
      CONSTRUCT(random);
#endif
      time[n] = timeFunctor(&random, means, timePerTest);
      meaninglessResult += random(means[0]);
   }


   {
      std::string name(OutputName);
      name += ".txt";
      std::ofstream out(name.c_str());
      for (int i = 0; i != upperBoundMeans.size(); ++i) {
         out << upperBoundMeans[i] << " " << time[i] << "\n";
      }
   }

   std::cout
         << "Number of distinct means = " << numberOfMeans << "\n"
         << "Multiplity of each mean = " << multiplicity << "\n"
         << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "Time given in nanoseconds.\n"
         << "Meaningless result = " << meaninglessResult << "\n";

   return 0;
}

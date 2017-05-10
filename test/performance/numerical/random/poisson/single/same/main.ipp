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

   // Get the mean.
   double mean = -1;
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the mean.\n";
      exit(1);
   }
   parser.getArgument(&mean);
   if (mean < 0) {
      std::cerr << "Error: Bad mean.\n";
      exit(1);
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

   double time;
   double meaninglessResult = 0;

#if defined(POISSON_SIZE_CONSTRUCTOR)
   CONSTRUCT(random, int(mean) + 1);
#elif defined(POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR)
   CONSTRUCT(random, normalThreshhold);
#elif defined(POISSON_MAX_MEAN_CONSTRUCTOR)
   CONSTRUCT(random, mean);
#else
   CONSTRUCT(random);
#endif
   time = timeFunctor(&random, mean, timePerTest);
   meaninglessResult += random(mean);

   std::cout << "Time = " << time << "\n"
             << "Mean = " << mean << "\n"
             << "# CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
             << "# Time given in nanoseconds.\n"
             << "# Meaningless result = " << meaninglessResult << "\n";

   return 0;
}

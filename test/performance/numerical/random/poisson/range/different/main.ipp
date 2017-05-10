// -*- C++ -*-

#ifndef __main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/array/Array.h"
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

   // Get the means.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the means file.\n";
      exit(1);
   }
   ads::Array<1, double> means;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> means;
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
   double normalThreshhold = 0;
   parser.getOption("threshhold", &normalThreshhold);
   if (normalThreshhold <= 0) {
      std::cerr << "Error: Bad normal threshhold.\n";
      exit(1);
   }
#endif

   // There should be no more options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   ads::Array<1, double> time(means.size());
   double meaninglessResult = 0;

   for (int i = 0; i != means.size(); ++i) {
      const double mean = means[i];
#if defined(POISSON_SIZE_CONSTRUCTOR)
      CONSTRUCT(random, int(mean) + 1);
#elif defined(POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR)
      CONSTRUCT(random, normalThreshhold);
#elif defined(POISSON_MAX_MEAN_CONSTRUCTOR)
      CONSTRUCT(random, mean);
#else
      CONSTRUCT(random);
#endif
      time[i] = timeFunctor(&random, mean, timePerTest);
      meaninglessResult += random(mean);
   }


   {
      std::string name(OutputName);
      name += ".txt";
      std::ofstream out(name.c_str());
      for (int i = 0; i != means.size(); ++i) {
         out << means[i] << " " << time[i] << "\n";
      }
   }

   std::cout
         << "# CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "# Time given in nanoseconds.\n"
         << "# Meaningless result = " << meaninglessResult << "\n";

   return 0;
}

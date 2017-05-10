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
#include "../timeFunctor.ipp"
#undef __timeFunctor_ipp__


int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the shape.
   double shape = -1;
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the shape.\n";
      exit(1);
   }
   parser.getArgument(&shape);
   if (shape < 0) {
      std::cerr << "Error: Bad shape.\n";
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

   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options.\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   double time;
   double meaninglessResult = 0;

   Gamma::DiscreteUniformGenerator uniform;
   Gamma::NormalGenerator normal(&uniform);
   Gamma random(&normal);
   time = timeFunctor(&random, shape, timePerTest);
   meaninglessResult += random(shape);

   std::cout << "Time = " << time << "\n"
             << "Shape = " << shape << "\n"
             << "# CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
             << "# Time given in nanoseconds.\n"
             << "# Meaningless result = " << meaninglessResult << "\n";

   return 0;
}

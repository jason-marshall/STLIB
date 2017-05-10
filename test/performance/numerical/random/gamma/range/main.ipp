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
#include "../timeFunctor.ipp"
#undef __timeFunctor_ipp__


int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Get the shapes.
   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Error: You must specify the shapes file.\n";
      exit(1);
   }
   ads::Array<1, double> shapes;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> shapes;
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

   // There should be no more options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error: Unmatched options:\n";
      parser.printOptions(std::cerr);
      exit(1);
   }

   ads::Array<1, double> time(shapes.size());
   double meaninglessResult = 0;

   Gamma::DiscreteUniformGenerator uniform;
   Gamma::NormalGenerator normal(&uniform);
   Gamma random(&normal);
   for (int i = 0; i != shapes.size(); ++i) {
      const double shape = shapes[i];
      time[i] = timeFunctor(&random, shape, timePerTest);
      meaninglessResult += random(shape);
   }


   {
      std::string name(OutputName);
      name += ".txt";
      std::ofstream out(name.c_str());
      for (int i = 0; i != shapes.size(); ++i) {
         out << shapes[i] << " " << time[i] << "\n";
      }
   }

   std::cout
         << "# CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
         << "# Time given in nanoseconds.\n"
         << "# Meaningless result = " << meaninglessResult << "\n";

   return 0;
}

// -*- C++ -*-

#ifndef __performance_numerical_random_exponential_offset_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <string>

using namespace stlib;

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " sum count\n";
   exit(1);
}

}

int
main(int argc, char* argv[]) {

   ads::ParseOptionsArguments parser(argc, argv);
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Get the sum.
   double sum = 0;
   if (! parser.getArgument(&sum)) {
      std::cerr << "Bad sum argument.\n";
      exitOnError();
   }

   // Get the number of deviates to generate.
   std::size_t count = 0;
   if (! parser.getArgument(&count)) {
      std::cerr << "Bad count argument.\n";
      exitOnError();
   }

   // Get the lower and upper bounds for output.
   double lower = 0;
   parser.getOption("lower", &lower);
   double upper = std::numeric_limits<double>::max();
   parser.getOption("upper", &upper);

   ExponentialGenerator::DiscreteUniformGenerator uniform;
   ExponentialGenerator random(&uniform);
   std::cout.precision(std::numeric_limits<double>::digits10);
   double previous = sum;
   double offset;
   for (std::size_t i = 0; i != count; ++i) {
      sum += random();
      offset = sum - previous;
      if (lower <= offset && offset <= upper) {
         std::cout << offset << '\n';
      }
      previous = sum;
   }

   return 0;
}

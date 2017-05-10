// -*- C++ -*-

/*!
  \file falseSharingArrayYes.cc
  \brief Test false sharing for OpenMP.
*/

#include "ads/timer.h"
#include "ads/utility.h"
#include "ads/array/Array.h"

#include <iostream>

#include <cassert>
#include <cmath>

#include <omp.h>

//
// Forward declarations.
//

//! Exit with an error message.
void
exitOnError();

//
// Global variables.
//

static std::string programName;

//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the reactions, input state and output state files.
   if (parser.getNumberOfArguments() != 0) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }
   // There should be no arguments.
   assert(parser.areArgumentsEmpty());

   //
   // Parse the options.
   //

#ifdef _OPENMP
   {
      int numberOfThreads = 0;
      if (parser.getOption("threads", &numberOfThreads)) {
         if (numberOfThreads < 1) {
            std::cerr << "Bad number of threads.\n";
            exitOnError();
         }
         omp_set_num_threads(numberOfThreads);
      }
   }
#endif

   // Number of steps
   int numberOfSteps = 1000;
   parser.getOption("steps", &numberOfSteps);

   // Array size.
   int arraySize = 10;
   parser.getOption("size", &arraySize);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Print information.
   //

   int numberOfThreads = 1;
#ifdef _OPENMP
   std::cout << "Number of processors = " << omp_get_num_procs() << "\n";
   {
#pragma omp parallel
      if (omp_get_thread_num() == 0) {
         numberOfThreads = omp_get_num_threads();
      }
      std::cout << "Number of threads = " << numberOfThreads << "\n";
   }
#else
   std::cout << "This is a sequential program.\n";
#endif
   std::cout << "Number of steps = " << numberOfSteps << "\n"
             << "Array size = " << arraySize << "\n";

   //
   // Run the test.
   //

   ads::Array<1, Number> data(numberOfThreads * arraySize);
   ads::Array<1, ads::Array<1, Number, false> > a(numberOfThreads);
   for (int i = 0; i != numberOfThreads; ++i) {
      a[i].rebuild(&data[i * arraySize], &data[(i + 1) * arraySize]);
      a[i] = 0;
   }

   ads::Timer timer;
   timer.tic();

#pragma omp parallel
   {
#ifdef _OPENMP
      const int ThreadNumber = omp_get_thread_num();
#else
      const int ThreadNumber = 0;
#endif
      ads::Array<1, Number, false>& x = a[ThreadNumber];
      for (int i = 0; i != numberOfSteps; ++i) {
         for (int j = 0; j != x.size(); ++j) {
            x[j] += sin(i + j);
         }
      }
   }

   Number sum = 0;
   for (int i = 0; i != numberOfThreads; ++i) {
      sum += ads::computeSum(a[i]);
   }

   double elapsedTime = timer.toc();
   std::cout << "Test time = " << elapsedTime << "\n"
             << "sum = \n" << sum << "\n";

   return 0;
}




void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-threads=t] [-steps=s]\n";
   exit(1);
}

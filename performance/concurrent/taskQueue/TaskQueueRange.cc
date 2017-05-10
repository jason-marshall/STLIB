// -*- C++ -*-

/*!
  \file TaskQueueRange.cc
  \brief Test the performance of the task queue.
*/

#include "concurrent/taskQueue/TaskQueueRange.h"
#include "ads/timer.h"
#include "ads/utility.h"

#include <iostream>
#include <iterator>
#include <vector>

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

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
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the number of tasks and the cost of a task.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   int numberOfTasks;
   if (! parser.getArgument(&numberOfTasks)) {
      std::cerr << "Error in reading the number of tasks.\n";
      exitOnError();
   }

   int cost;
   if (! parser.getArgument(&cost)) {
      std::cerr << "Error in reading the number of tasks.\n";
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
      if (parser.getOption("threads", &numberOfThreads) ||
            parser.getOption("t", &numberOfThreads)) {
         if (numberOfThreads < 1) {
            std::cerr << "Bad number of threads.\n";
            exitOnError();
         }
         omp_set_num_threads(numberOfThreads);
      }
   }
#endif

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
   std::cout << "This is a serial program.\n";
#endif
   std::cout << "Number of tasks = " << numberOfTasks << "\n";

   //
   // Run the test.
   //

   concurrent::TaskQueueRange<> taskQueue(0, numberOfTasks);

   ads::Timer timer;
   timer.tic();

   double sum = 0;
   std::vector<int> tasks(numberOfThreads);
#pragma omp parallel reduction(+ : sum)
   {
      int count = 0, task;
      double t;
      if (cost == 0) {
         while ((task = taskQueue.pop()) != taskQueue.getEnd()) {
            ++count;
            sum += task;
         }
      }
      else {
         while ((task = taskQueue.pop()) != taskQueue.getEnd()) {
            ++count;
            t = task;
            for (int i = 0; i != cost; ++i) {
               t = sin(t);
            }
            sum += t;
         }
      }
#ifdef _OPENMP
      tasks[omp_get_thread_num()] = count;
#else
      tasks[0] = count;
#endif
   }
   double elapsedTime = timer.toc() / numberOfThreads;

   std::cout << "Meaningless result = " << sum
             << "\nCost per task = " << cost
             << "\nTask counts = ";
   std::copy(tasks.begin(), tasks.end(),
             std::ostream_iterator<int>(std::cout, " "));
   std::cout << "\nTest time = " << elapsedTime << "\n";

   const int minimumTasks = *std::min_element(tasks.begin(), tasks.end());
   if (minimumTasks != 0) {
      std::cout << "Time per task (while sharing) = "
                << elapsedTime / minimumTasks * 1e6
                << " milliseconds.\n";
   }
   else {
      std::cout << "The tasks were not shared.\n";
   }

   return 0;
}

void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-threads=t] tasks cost\n";
   exit(1);
}

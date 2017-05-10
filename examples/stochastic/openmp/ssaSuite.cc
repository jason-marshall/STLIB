// -*- C++ -*-

/*!
  \file ssaSuite.cc
  \brief Stochastic simulations.
*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page ssaSuiteDriver The Stochastic Simulation Driver

<!--------------------------------------------------------------------------->
\section ssaSuiteDriverIntroduction Introduction


<!--------------------------------------------------------------------------->
\section ssaSuiteDriverUsage Usage

\verbatim
ssaSuite.exe [-tau] [-threads=t] [-epsilon=e] [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]
  reactions inputState outputState
\endverbatim


<!--------------------------------------------------------------------------->
\section ssaSuiteDriverExamples Examples

*/

#ifndef _OPENMP
#error You must compile this program with OpenMP enabled.
#endif

// Get the sequential code for stochastic simulations.
// We disable OpenMP to include the code and then re-enable it so we will
// be able to spawn multiple threads running the serial algorithm.
#define TEMP_OPENMP _OPENMP
#undef _OPENMP
#include "stlib/stochastic/direct.h"
#include "stlib/stochastic/TauLeaping.h"
#define _OPENMP TEMP_OPENMP
//#define _OPENMP

#ifndef _OPENMP
// CONTINUE: This causes an error when generating the dependencies.
#error You must compile this program with OpenMP enabled.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "numerical/random/UniformRandom.h"
#include "numerical/random/PoissonDeviateInversionRejection.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <cassert>

#include <omp.h>

namespace {

//
// Global variables.
//

//! The program name.
std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n"
         << " [-tau] [-threads=t] [-epsilon=e]\n"
         << " [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]\n"
         << " reactions inputState outputState\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;
   typedef numerical::UniformRandom2<Number> UniformRandom;
   typedef numerical::PoissonDeviateInversionRejection
   <Number, numerical::UniformRandom2> Poisson;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the reactions, input state and output state files.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Parse the options.
   //

   // If we are using tau-leaping.
   const bool areUsingTauLeaping = parser.getOption("tau");

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

   // The allowed error in the tau-leaping method.
   Number epsilon = 0;
   if (areUsingTauLeaping) {
      parser.getOption("epsilon", &epsilon);
      if (epsilon <= 0) {
         std::cerr << "Error: Bad value for epsilon.\n";
         exitOnError();
      }
   }

   // Starting time
   Number startTime = 0;
   parser.getOption("startTime", &startTime);

   // Ending time
   Number endTime = std::numeric_limits<Number>::max();
   parser.getOption("endTime", &endTime);

   // Number of steps
   int numberOfSteps = 0;
   parser.getOption("steps", &numberOfSteps);

   if (endTime == std::numeric_limits<Number>::max() && numberOfSteps == 0) {
      std::cerr << "Error: You must specify either the ending time or the\n"
                << "number of steps.\n";
      exitOnError();
   }

   // Seed for the random number generator.
   int seed = 1;
   parser.getOption("seed", &seed);
   if (seed == 0) {
      std::cerr << "Error: The seed must be nonzero.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Read in the reactions and the initial state.
   //

   // Construct the state classes.
   ads::Array<1, stochastic::State<Number> > states(omp_get_num_threads());

   // Read the reactions.
   {
      const std::string fileName = parser.getArgument();
      for (int i = 0; i != states.size(); ++i) {
         std::ifstream in(fileName.c_str());
         stochastic::readReactionsAscii(in, &states[i]);
      }
   }

   // Read the initial populations.
   {
      const std::string fileName = parser.getArgument();
      for (int i = 0; i != states.size(); ++i) {
         std::ifstream in(fileName.c_str());
         stochastic::readPopulationsAscii(in, &states[i]);
      }
   }

   // Check the validity of the initial state.
   if (! isValid(states[0])) {
      std::cerr << "Error: The initial state of the simulation is not valid.\n";
      exitOnError();
   }

   //
   // Print information.
   //

   std::cout << "Number of processors = " << omp_get_num_procs() << "\n";
   {
      int numberOfThreads = 0;
#pragma omp parallel
      if (omp_get_thread_num() == 0) {
         numberOfThreads = omp_get_num_threads();
      }
      std::cout << "Number of threads = " << numberOfThreads << "\n";
   }
   std::cout << "Number of reactions = "
             << states[0].getNumberOfReactions() << "\n"
             << "Number of species = "
             << states[0].getNumberOfSpecies() << "\n";

   //
   // Run the simulation.
   //

   ads::Array<1, int> simulationSteps(omp_get_num_threads(), 0);

   std::cout << "Running the simulation...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // If they specified the ending time.
   if (endTime != std::numeric_limits<Number>::max()) {
      if (areUsingTauLeaping) {
#pragma omp parallel
         {
            const int n = omp_get_thread_num();
            simulationSteps[n] =
            stochastic::simulateWithTauLeaping<Poisson>(&states[n], epsilon,
            endTime, seed + n);
         }
      }
      else {
#pragma omp parallel
         {
            const int n = omp_get_thread_num();
            simulationSteps[n] =
            stochastic::computeDirectSsa<UniformRandom>(&states[n], endTime,
            seed + n);
         }
      }
   }
   // Otherwise they specified the number of steps.
   else {
      if (areUsingTauLeaping) {
#pragma omp parallel
         {
            const int n = omp_get_thread_num();
            stochastic::simulateWithTauLeaping<Poisson>(&states[n], epsilon,
            numberOfSteps, seed + n);
            simulationSteps[n] = numberOfSteps;
         }
      }
      else {
#pragma omp parallel
         {
            const int n = omp_get_thread_num();
            stochastic::computeDirectSsa<UniformRandom>(&states[n], numberOfSteps,
            seed + n);
            simulationSteps[n] = numberOfSteps;
         }
      }
   }

   double elapsedTime = timer.toc();
   std::cout << "Done.  Simulation time = " << elapsedTime
             << "\n" << std::flush;
   std::cout << "The simulation took " << simulationSteps << " steps.\n"
             << "The simulation time interval is [" << startTime
             << " .. " << states[0].getTime() << "].\n";


   // Write the output states.
   {
      const std::string baseName = parser.getArgument();
      for (int i = 0; i != states.size(); ++i) {
         std::ostringstream name;
         name << baseName << i;
         std::ofstream out(name.str().c_str());
         out << states[i].getPopulations();
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}

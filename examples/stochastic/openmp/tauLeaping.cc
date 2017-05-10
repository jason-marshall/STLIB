// -*- C++ -*-

/*!
  \file ssa.cc
  \brief Tau leaping.
*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page ssaDriver The Stochastic Simulation Driver

<!--------------------------------------------------------------------------->
\section ssaDriverIntroduction Introduction


<!--------------------------------------------------------------------------->
\section ssaDriverUsage Usage

\verbatim
ssa.exe [-threads=t] [-small={0,1}] [-epsilon=e] [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]
  reactions inputState outputState
\endverbatim


<!--------------------------------------------------------------------------->
\section ssaDriverExamples Examples

*/

#ifdef _OPENMP
#include "stlib/stochastic/TauLeapingThreaded.h"
#else
#include "stlib/stochastic/TauLeapingSerial.h"
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "numerical/random/poisson/PoissonGeneratorInvIfmAcNorm.h"

#include <iostream>
#include <fstream>

#include <cassert>

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
         << " [-threads=t] [-small={0,1}] [-epsilon=e]\n"
         << " [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]\n"
         << " reactions inputState outputState\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;
   typedef numerical::PoissonGeneratorInvIfmAcNorm<Number> PoissonGenerator;
   typedef PoissonGenerator::DiscreteUniformGenerator DiscreteUniformGenerator;
   typedef PoissonGenerator::NormalGenerator NormalGenerator;

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

#ifdef _OPENMP
   // If this is a small problem.
   bool isSmall = false;
   {
      int value;
      if (parser.getOption("small", &value)) {
         isSmall = value;
      }
   }

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

   // The allowed error in the tau-leaping method.
   Number epsilon = 0;
   parser.getOption("epsilon", &epsilon);
   if (epsilon <= 0) {
      std::cerr << "Error: Bad value for epsilon.\n";
      exitOnError();
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
   DiscreteUniformGenerator::result_type seed = 1;
   if (parser.getOption("seed", &seed)) {
      // CONTINUE: Seed the generator here.
      if (seed == 0) {
         std::cerr << "Error: The seed must be nonzero.\n";
         exitOnError();
      }
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

   // Construct the state class.
   stochastic::State<Number> state;

   // Read the reactions.
   {
      std::ifstream in(parser.getArgument().c_str());
      stochastic::readReactionsAscii(in, &state);
   }

   // Read the initial populations.
   {
      std::ifstream in(parser.getArgument().c_str());
      stochastic::readPopulationsAscii(in, &state);
   }

   // Check the validity of the initial state.
   if (! isValid(state)) {
      std::cerr << "Error: The initial state of the simulation is not valid.\n";
      exitOnError();
   }

#ifdef _OPENMP
   // If this should be treated as a small problem.
   if (isSmall) {
      // Set the threshholds so it is considered small.
      state.setSmallNumberOfReactions(state.getNumberOfReactions());
      state.setSmallNumberOfSpecies(state.getNumberOfSpecies());
   }
   else {
      // Set the threshholds so it is considered large.
      state.setSmallNumberOfReactions(0);
      state.setSmallNumberOfSpecies(0);
   }
#endif

   //
   // Print information.
   //

#ifdef _OPENMP
   std::cout << "Number of processors = " << omp_get_num_procs() << "\n";
   {
      int numberOfThreads = 0;
#pragma omp parallel
      if (omp_get_thread_num() == 0) {
         numberOfThreads = omp_get_num_threads();
      }
      std::cout << "Number of threads = " << numberOfThreads << "\n";
   }
#else
   std::cout << "This is a sequential program.\n";
#endif
   std::cout << "Number of reactions = " << state.getNumberOfReactions() << "\n"
             << "Number of species = " << state.getNumberOfSpecies() << "\n";

   //
   // Run the simulation.
   //

   int simulationSteps = 0;

   std::cout << "Running the simulation...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // If they specified the ending time.
   if (endTime != std::numeric_limits<Number>::max()) {
#ifdef _OPENMP
      simulationSteps =
         stochastic::simulateWithTauLeapingThreaded<Poisson>
         (&state, epsilon, endTime, seed);
#else
      simulationSteps =
         stochastic::simulateWithTauLeapingSerial<Poisson>
         (&state, epsilon, endTime, seed);
#endif
   }
   // Otherwise they specified the number of steps.
   else {
#ifdef _OPENMP
      stochastic::simulateWithTauLeapingThreaded<Poisson>
      (&state, epsilon, numberOfSteps, seed);
#else
      stochastic::simulateWithTauLeapingSerial<Poisson>
      (&state, epsilon, numberOfSteps, seed);
#endif
      simulationSteps = numberOfSteps;
   }

   double elapsedTime = timer.toc();
   std::cout << "Done.  Simulation time = " << elapsedTime
             << "\n" << std::flush;
   std::cout << "The simulation took " << simulationSteps << " steps.\n"
             << "The simulation time interval is [" << startTime
             << " .. " << state.getTime() << "].\n";


   // Write the output state.
   {
      std::ofstream out(parser.getArgument().c_str());
      out << state.getPopulations();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}

// -*- C++ -*-

/*!
  \file tauLeaping.cc
  \brief Tau leaping.
*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page tauLeapingDriver The Tau Leaping Driver

<!--------------------------------------------------------------------------->
\section tauLeapingDriverUsage Usage

\verbatim
tauLeaping.exe [-epsilon=e] [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]
  [-n=ensembleSize] [-o=output]
  reactions rateConstants populations
\endverbatim


<!--------------------------------------------------------------------------->
\section ssaDriverExamples Examples

*/

// CONTINUE
// 30% improvement.
#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
// 10% improvement.
#define NUMERICAL_POISSON_STORE_INVERSE

#include "stlib/stochastic/TauLeaping.h"
//#include "stlib/stochastic/ReactionSet.h"

#include "stlib/ads/array/SparseArray.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

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
         << " [-epsilon=number] [-startTime=number] [-endTime=number] [-steps=integer]\n"
         << " [-seed=integer] [-n=integer] [-o=filename]"
         << "\n reactions rateConstants populations\n"
         << "You must specify a value for epsilon, the allowed error.\n"
         << "Specify the ensemble size with -n and the output file with -o."
         << "You must specify either a rate to use for all reactions or a rate\n"
         << "constants file.  Likewise for the populations.\n";
   exit(1);
}

}

//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;

   // Use signed integers to store the populations.
   typedef stochastic::State<ads::Array<1, std::ptrdiff_t> > State;

   typedef stochastic::TauLeaping<State> TauLeaping;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the reactions and the input state files.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Parse the options.
   //

   // The allowed error in the tau-leaping method.
   Number epsilon = 0;
   parser.getOption("epsilon", &epsilon);
   if (epsilon <= 0) {
      std::cerr << "Error: Bad value for epsilon.\n";
      exitOnError();
   }

   //
   // The termination condition.
   //

   // Starting time
   Number startTime = 0;
   parser.getOption("startTime", &startTime);

   // Ending time
   Number endTime = std::numeric_limits<Number>::max();
   parser.getOption("endTime", &endTime);

   // Number of steps
   unsigned long numberOfSteps = std::numeric_limits<unsigned long>::max();
   parser.getOption("steps", &numberOfSteps);

   if (endTime == std::numeric_limits<Number>::max() &&
         numberOfSteps == std::numeric_limits<unsigned long>::max()) {
      std::cerr << "Error: You must specify either the ending time or the\n"
                << "number of steps.\n";
      exitOnError();
   }

   // CONTINUE
   stochastic::EssTerminationConditionEndTimeReactionCount<Number>
   terminationCondition(endTime, numberOfSteps);

   // Seed for the random number generator.
   TauLeaping::DiscreteUniformGenerator::result_type seed = 0;
   parser.getOption("seed", &seed);

   //
   // Read in the reactions and the initial state.
   //
   std::cout << "Reading the reactions and the initial state.\n";
   ads::Timer timer;
   timer.tic();

   // Construct the reaction set.
   stochastic::ReactionSet<Number> reactions;

   // Read the reactants and products for the reactions.
   {
      std::ifstream in(parser.getArgument().c_str());
      stochastic::readReactantsAndProductsAscii(in, &reactions);
   }

   // Read the rate constants for the reactions.
   {
      std::istringstream argument(parser.getArgument());
      Number rate = -1;
      argument >> rate;
      // If they specified a numeric value.
      if (rate != -1) {
         reactions.setRateConstants(rate);
      }
      // If they gave a file.
      else {
         std::ifstream in(argument.str().c_str());
         stochastic::readRateConstantsAscii(in, &reactions);
      }
   }

   // Read the initial populations.
   State::PopulationsContainer initialPopulations;
   {
      std::istringstream argument(parser.getArgument());
      int population = -1;
      argument >> population;
      // If they specified a numeric value.
      if (population != -1) {
         initialPopulations.resize(reactions.computeNumberOfSpecies());
         initialPopulations = population;
      }
      // If they gave a file.
      else {
         std::ifstream in(argument.str().c_str());
         stochastic::readPopulations(in, &initialPopulations);
      }
   }
   double elapsedTime = timer.toc();
   std::cout << "Done. Elapsed time = " << elapsedTime << '\n';

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   //
   // Build the state change vectors.
   //

   std::cout << "Building the state change vectors...\n";
   timer.tic();
   State::ScvContainer stateChangeVectors;
   stochastic::buildStateChangeVectors
   (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
    &stateChangeVectors);
   elapsedTime = timer.toc();
   std::cout << "Done. Elapsed time = " << elapsedTime << '\n';

   //
   // The output.
   //

   std::string outputName;
   parser.getOption("o", &outputName);
   std::ofstream outputStream;
   if (! outputName.empty()) {
      outputStream.open(outputName.c_str());
   }

   //
   // The ensemble size.
   //

   int ensembleSize = 1;
   if (parser.getOption("n", &ensembleSize)) {
      if (ensembleSize < 1) {
         std::cerr << "Error: Bad ensemble size of " << ensembleSize << "\n";
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
   // Print information.
   //

   std::cout << "Number of reactions = "
             << reactions.getSize() << "\n"
             << "Number of species = "
             << initialPopulations.size() << "\n"
             << "Ensemble size = " << ensembleSize << "\n";

   unsigned long reactionCount = 0, stepCount = 0;

   // Construct the state class.
   State state(initialPopulations, stateChangeVectors);

   //
   // Construct the simulation class.
   //
   TauLeaping tauLeaping(state, reactions, epsilon);

   // Seed the random number generator.
   tauLeaping.seed(seed);

   //
   // Run the simulation.
   //

   std::cout << "Running the simulations...\n" << std::flush;

   timer.tic();
   for (int n = 0; n != ensembleSize; ++n) {
      tauLeaping.initialize(initialPopulations, startTime);
      tauLeaping.simulate(terminationCondition);
      reactionCount += tauLeaping.getState().getReactionCount();
      stepCount += tauLeaping.getStepCount();

      // Write the final state.
      if (! outputName.empty()) {
         outputStream << tauLeaping.getState().getTime() << "\n"
                      << tauLeaping.getState().getPopulations() << "\n";
      }
   }
   elapsedTime = timer.toc();

   std::cout << "Done.  Simulation time = " << elapsedTime << "\n"
             << "The ensemble of " << ensembleSize << " simulations took "
             << stepCount << " steps.\n"
             << "Number of reactions = " << reactionCount << ".\n"
             << "Reactions per second = "
             << reactionCount / elapsedTime << ".\n"
             << "Time per reaction = " << elapsedTime / reactionCount * 1e9
             << " nanoseconds\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout << elapsedTime / reactionCount * 1e9  << "\n";

   return 0;
}

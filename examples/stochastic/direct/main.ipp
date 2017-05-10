// -*- C++ -*-

/*!
  \file examples/stochastic/direct/main.ipp
  \brief Exact stochastic simulations with the direct method.
*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page stochasticDirect Exact stochastic simulations with the sum-of-propensities method.

<!--------------------------------------------------------------------------->
\section stochasticDirectIntroduction Introduction


<!--------------------------------------------------------------------------->
\section stochasticDirectUsage Usage

\verbatim
programName.exe [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]
  [-n=ensembleSize] [-repair=m] [-rebuild=n] [-o=output]
  reactions rateConstants populations
\endverbatim

\note Not all data structures support the repair and rebuild options. (Not all
of them need repairing or rebuilding.)

<!--------------------------------------------------------------------------->
\section stochasticDirectExamples Examples

*/

#ifndef __stochastic_direct_main_ipp__
#error This file is an implementation detail.
#endif

#include "stochastic/Direct.h"
#include "stochastic/ReactionSet.h"
#include "stochastic/reactionPropensityInfluence.h"
#include "stochastic/Propensities.h"
#include "stochastic/PropensitiesDecayingDimerizing.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "ext/vector.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <cassert>

namespace {

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n"
         << " [-startTime=s] [-endTime=e] [-steps=s] [-seed=s] [-n=ensembleSize] [-o=output]"
#ifdef STOCHASTIC_SOP_REPAIR
         << " [-repair=m]"
#endif
#ifdef STOCHASTIC_SOP_REBUILD
         << " [-rebuild=n]"
#endif
         << "\n reactions rateConstants populations\n"
         << "You must specify either a rate to use for all reactions or a rate\n"
         << "constants file.  Likewise for the populations.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;
   typedef ExponentialGenerator::DiscreteUniformGenerator
   DiscreteUniformGenerator;

   typedef stochastic::State<StateChangeArray> State;

#ifdef STOCHASTIC_SOP_USE_INFLUENCE
   // If we use the reaction influence array, we will compute the propensities
   // one at a time.
   typedef stochastic::PropensitiesSingle<Number> PropensitiesFunctor;
#else
   // Otherwise, we compute the propensities all at once.
   typedef stochastic::PropensitiesAll<Number> PropensitiesFunctor;
#endif

   typedef stochastic::Direct < DiscreteFiniteGenerator, ExponentialGenerator,
           PropensitiesFunctor, State > Direct;

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

   stochastic::EssTerminationConditionEndTimeReactionCount<Number>
   terminationCondition(endTime, numberOfSteps);

   // Seed for the random number generator.
   DiscreteUniformGenerator::result_type seed = 0;
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
#ifndef STOCHASTIC_DIRECT_FIXED
         initialPopulations.resize(reactions.computeNumberOfSpecies());
#endif
         std::fill(initialPopulations.begin(), initialPopulations.end(),
                   population);
      }
      // If they gave a file.
      else {
         std::ifstream in(argument.str().c_str());
         in >> initialPopulations;
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
   // Build the array of reaction influences.
   //

   // If not used, this is left empty.
   ads::StaticArrayOfArrays<int> reactionInfluence;
#ifdef STOCHASTIC_SOP_USE_INFLUENCE
   std::cout << "Building the reaction influence data structure...\n";
   timer.tic();
   stochastic::computeReactionPropensityInfluence
   (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
    &reactionInfluence, true);
   elapsedTime = timer.toc();
   std::cout << "Done. Elapsed time = " << elapsedTime << '\n';
#endif

   //
   // The propensities functor.
   //

   PropensitiesFunctor propensitiesFunctor(reactions);

   //
   // The propensities.
   //

   // CONTINUE
#if 0
   stochastic::PropensitiesDecayingDimerizing<Number> propensities;
#endif

   //
   // Get more repairing and rebuilding options.
   //

#ifdef STOCHASTIC_SOP_REPAIR
   // For loosely coupled problems, you can specify how many steps to take
   // between repairs.
   int stepsBetweenRepairs = 0;
   if (parser.getOption("repair", &stepsBetweenRepairs)) {
      if (stepsBetweenRepairs < 1) {
         std::cerr
               << "Error: Bad value for the number of steps between repairs.\n";
         exitOnError();
      }
   }
#endif

#ifdef STOCHASTIC_SOP_REBUILD
   // The number of steps to take between rebuilds.
   int stepsBetweenRebuilds = 0;
   if (parser.getOption("rebuild", &stepsBetweenRebuilds)) {
      if (stepsBetweenRebuilds < 1) {
         std::cerr
               << "Error: Bad value for the number of steps between rebuilds.\n";
         exitOnError();
      }
   }
#endif

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

#ifdef STOCHASTIC_SOP_USE_INFLUENCE
   std::cout << "This method uses the reaction influence array.\n";
#else
   std::cout << "This method does not use the reaction influence array.\n";
#endif

#ifdef STOCHASTIC_SOP_REPAIR
   if (stepsBetweenRepairs != 0) {
      std::cout << "Number of steps between repairs = "
                << stepsBetweenRepairs << "\n";
   }
#endif

#ifdef STOCHASTIC_SOP_REBUILD
   if (stepsBetweenRebuilds != 0) {
      std::cout << "Number of steps between rebuilds = "
                << stepsBetweenRebuilds << "\n";
   }
#endif

   unsigned long reactionCount = 0;

   std::cout << "Running the simulations...\n" << std::flush;

   // Construct the state class.
   State state(initialPopulations, stateChangeVectors);

   // CONTINUE
#if 0
   // Check the validity of the initial state.
   if (! state.isValid()) {
      std::cerr << "Error: The initial state of the simulation is not valid.\n";
      exitOnError();
   }
#endif

   //
   // Construct the simulation class.
   //
   Direct direct(state, propensitiesFunctor, reactionInfluence);

   // Seed the random number generator.
   direct.seed(seed);

   //
   // Configure the random number generators.
   //

#ifdef STOCHASTIC_SOP_USE_INFLUENCE_IN_GENERATOR
   direct.getDiscreteFiniteGenerator().setInfluence(&reactionInfluence);
#endif

#ifdef STOCHASTIC_SOP_REPAIR
   if (stepsBetweenRepairs != 0) {
      direct.getDiscreteFiniteGenerator().
      setStepsBetweenRepairs(stepsBetweenRepairs);
   }
#endif

#ifdef STOCHASTIC_SOP_REBUILD
   if (stepsBetweenRebuilds != 0) {
      direct.getDiscreteFiniteGenerator().
      setStepsBetweenRebuilds(stepsBetweenRebuilds);
   }
#endif

   //
   // Run the simulation.
   //

   timer.tic();
   for (int n = 0; n != ensembleSize; ++n) {
      direct.initialize(initialPopulations, startTime);
      direct.simulate(terminationCondition);
      reactionCount += direct.getState().getReactionCount();

      // Write the final state.
      if (! outputName.empty()) {
         outputStream << direct.getState().getTime() << "\n"
                      << direct.getState().getPopulations() << "\n";
      }
   }
   elapsedTime = timer.toc();

   std::cout << "Done.  Simulation time = " << elapsedTime << "\n"
             << "The ensemble of " << ensembleSize << " simulations took "
             << reactionCount << " steps.\n"
             << "Reactions per second = "
             << reactionCount / elapsedTime << ".\n"
             << "Time per reaction = " << elapsedTime / reactionCount * 1e9
             << " nanoseconds\n";
   std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
   std::cout.precision(0);
   std::cout << elapsedTime / reactionCount * 1e9  << "\n";

   return 0;
}

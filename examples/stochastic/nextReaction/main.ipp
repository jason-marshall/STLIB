// -*- C++ -*-

/*!
  \file examples/stochastic/nextReaction/main.ipp
  \brief Exact stochastic simulations with the next reaction method.
*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page stochasticNextReaction Exact stochastic simulations with the next reaction method.

<!--------------------------------------------------------------------------->
\section stochasticNextReactionIntroduction Introduction


<!--------------------------------------------------------------------------->
\section stochasticNextReactionUsage Usage

<!--[-threads=t] [-small={0,1}]-->
\verbatim
programName.exe [-startTime=s] [-endTime=e] [-steps=s] [-seed=s]
  [-n=ensembleSize] [-o=output]
  reactions rateConstants populations
\endverbatim

<!--------------------------------------------------------------------------->
\section stochasticNextReactionExamples Examples

*/

#ifndef __stochastic_nextReaction_main_ipp__
#error This file is an implementation detail.
#endif

#include "stochastic/NextReaction.h"
#include "stochastic/ReactionPriorityQueue.h"
#include "stochastic/ReactionSet.h"
#include "stochastic/reactionPropensityInfluence.h"
#include "stochastic/Propensities.h"
#include "stochastic/EssTerminationCondition.h"

//#define DECAYING_DIMERIZING
#ifdef DECAYING_DIMERIZING
#include "stochastic/PropensitiesDecayingDimerizing.h"
#endif
// CONTINUE
#define PROPENSITIES_SINGLE
//#define PROPENSITIES_ALL

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#ifdef STOCHOSTIC_NEXTREACTION_INVERSION
#include "numerical/random/exponential/ExponentialGeneratorInversion.h"
#endif

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
         << " [-startTime=real] [-endTime=real] [-steps=integer] [-seed=integer]\n"
         << " [-n=integer] [-o=name] "
#ifdef STOCHASTIC_NEXTREACTION_HASHING
         << " [-table=integer] [-load=real]"
#endif
#ifdef BALANCE_COSTS
         << " [-cost=real]"
#endif
         << "\n reactions rateConstants populations\n"
         << "-n specifies the ensemble size.\n"
         << "-o specifies the output file.\n"
#ifdef STOCHASTIC_NEXTREACTION_HASHING
         << "-table specifies the size for the hash table.\n"
         << "-load specifies the target load for the hash table.\n"
#endif
         << "You must specify either a rate to use for all reactions or a rate\n"
         << "constants file.  Likewise for the populations.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   //
   // Types.
   //

   typedef double Number;
   // The indexed priority queue is defined in the .cc file.
#if defined(STOCHOSTIC_NEXTREACTION_USE_INFINITY)
   typedef stochastic::ReactionPriorityQueue < IndexedPriorityQueue,
           numerical::EXPONENTIAL_GENERATOR_DEFAULT<>, true >
           ReactionPriorityQueue;
#elif defined(STOCHOSTIC_NEXTREACTION_INVERSION)
   // Inversion method for the exponential generator.
   typedef stochastic::ReactionPriorityQueue < IndexedPriorityQueue,
           numerical::ExponentialGeneratorInversion<> >
           ReactionPriorityQueue;
#else
   // Use the default exponential generator.
   typedef stochastic::ReactionPriorityQueue<IndexedPriorityQueue>
   ReactionPriorityQueue;
#endif
   typedef ReactionPriorityQueue::DiscreteUniformGenerator
   DiscreteUniformGenerator;

   // Dense population array.
   typedef ads::Array<1, int> PopulationArray;
   // Sparse state change vector.
   typedef ads::SparseArray<1, int> StateChangeArray;

   // The State.
   typedef stochastic::State < PopulationArray,
           ads::Array<1, StateChangeArray> > State;

   // The propensities functor.
#if defined(PROPENSITIES_SINGLE)

#ifdef DECAYING_DIMERIZING
   typedef stochastic::PropensitiesSingleDecayingDimerizing<Number>
   PropensitiesFunctor;
#else
   typedef stochastic::PropensitiesSingle<Number> PropensitiesFunctor;
#endif

#elif defined(PROPENSITIES_ALL)

#ifdef DECAYING_DIMERIZING
   typedef stochastic::PropensitiesAllDecayingDimerizing<Number>
   PropensitiesFunctor;
#else
   typedef stochastic::PropensitiesAll<Number> PropensitiesFunctor;
#endif

#else
#error The method of computing propensities has not been specified.    
#endif

   // The simulation class.
   typedef stochastic::NextReaction < State, PropensitiesFunctor,
           ReactionPriorityQueue > NextReaction;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

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

#ifdef STOCHASTIC_NEXTREACTION_HASHING
   int tableSize = std::max(256, 2 * reactions.getSize());
   parser.getOption("table", &tableSize);
   if (tableSize < 1) {
      std::cerr << "Bad size for the hash table = " << tableSize << "\n";
      exitOnError();
   }

   Number targetLoad = 2;
   parser.getOption("load", &targetLoad);
   if (targetLoad <= 0) {
      std::cerr << "Bad target load for the hash table = " << targetLoad << "\n";
      exitOnError();
   }
#endif

#ifdef BALANCE_COSTS
   Number cost = 0;
   if (parser.getOption("cost", &cost)) {
      if (cost <= 0) {
         std::cerr << "Bad cost constant = " << cost << "\n";
         exitOnError();
      }
   }
#endif

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
   // Construct the state class.
   //

   State state(initialPopulations, stateChangeVectors);

   //
   // Build the array of reaction influences.
   //

   std::cout << "Building the reaction influence data structure...\n";
   timer.tic();
   ads::StaticArrayOfArrays<int> reactionInfluence;
   stochastic::computeReactionPropensityInfluence
   (state.getNumberOfSpecies(), reactions.getBeginning(), reactions.getEnd(),
    &reactionInfluence, false);
   elapsedTime = timer.toc();
   std::cout << "Done. Elapsed time = " << elapsedTime << '\n';

   //
   // The propensities functor.
   //

   // CONTINUE
#ifdef DECAYING_DIMERIZING
   PropensitiesFunctor propensitiesFunctor;
#else
   PropensitiesFunctor propensitiesFunctor(reactions);
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

   double reactionCount = 0;

   std::cout << "Running the simulations...\n" << std::flush;

   // The discrete uniform generator.
   DiscreteUniformGenerator discreteUniformGenerator;
   // CONTINUE
   discreteUniformGenerator.seed(seed);

   // The reaction priority queue.
#if defined(STOCHASTIC_NEXTREACTION_HASHING)
   ReactionPriorityQueue reactionPriorityQueue(state.getNumberOfReactions(),
         &discreteUniformGenerator,
         tableSize, targetLoad);

#else
   ReactionPriorityQueue reactionPriorityQueue(state.getNumberOfReactions(),
         &discreteUniformGenerator);
#endif

#ifdef BALANCE_COSTS
   // If they specified a cost-balancing constant.
   if (cost != 0) {
      reactionPriorityQueue.setCostConstant(cost);
   }
#endif

   // Construct the simulation class.
   NextReaction nextReaction(state, propensitiesFunctor,
                             &reactionPriorityQueue, reactionInfluence);

   //
   // Run the simulation.
   //

   elapsedTime = 0;
   for (int n = 0; n != ensembleSize; ++n) {
      timer.tic();
      nextReaction.initialize(initialPopulations, startTime);
      nextReaction.simulate(terminationCondition);
      reactionCount += nextReaction.getState().getReactionCount();
      elapsedTime += timer.toc();

      // Write the final state.
      if (! outputName.empty()) {
         outputStream << nextReaction.getState().getTime() << "\n"
                      << nextReaction.getState().getPopulations() << "\n";
      }
   }

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

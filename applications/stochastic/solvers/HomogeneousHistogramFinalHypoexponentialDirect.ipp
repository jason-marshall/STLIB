// -*- C++ -*-

#ifndef __HomogeneousHistogramFinalHypoexponentialDirect_ipp__
#error This file is an implementation detail.
#endif

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
#include "Propensities.h"
#endif

#include "stochastic/HistogramFinalHypoexponentialDirect.h"
#include "stochastic/ReactionSet.h"
#include "stochastic/reactionPropensityInfluence.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <iterator>

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
      << programName << "\n\n"
      << "This program reads the model and simulations parameters from\n"
      << "stdin and writes the histogram to stdout.\n";
    // CONTINUE
    exit(1);
  }

}


//! The main loop.
int
main(int argc, char* argv[]) {
  typedef DiscreteGenerator::DiscreteUniformGenerator DiscreteUniformGenerator;

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
  typedef Propensities<true> PropensitiesFunctor;
#else
  // If we use the reaction influence array, we will compute the propensities
  // one at a time.
  typedef stochastic::PropensitiesSingle<true> PropensitiesFunctor;
#endif

  typedef PropensitiesFunctor::ReactionSetType ReactionSet;
  typedef stochastic::HistogramFinalHypoexponentialDirect<DiscreteGenerator,
    PropensitiesFunctor> Solver;

  typedef stochastic::State State;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE: Error message.
  assert(solverParameters.size() == 0);

  // Number of frames and frame times.
  std::vector<double> frameTimes;
  std::cin >> frameTimes;
  // Only one frame is allowed.
  assert(frameTimes.size() == 1);

  // Number of bins in the histograms.
  std::size_t numberOfBins;
  std::cin >> numberOfBins;

  // Histogram multiplicity.
  std::size_t histogramMultiplicity;
  std::cin >> histogramMultiplicity;

  //
  // Build the array of reaction influences.
  //
  array::StaticArrayOfArrays<std::size_t> reactionInfluence;
  stochastic::computeReactionPropensityInfluence
    (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
     &reactionInfluence, true);

  //
  // Construct the simulation class.
  //
  Solver solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()), 
		PropensitiesFunctor(reactions), reactionInfluence,
		frameTimes, recordedSpecies, numberOfBins,
		histogramMultiplicity, maximumAllowedSteps);

  //
  // Read the Mersenne twister state.
  //
  std::cin >> solver.getDiscreteUniformGenerator();

#ifdef STOCHASTIC_USE_INFLUENCE_IN_GENERATOR
  solver.getDiscreteGenerator().setInfluence(&reactionInfluence);
#endif

  // There should be no more options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  //
  // Run the simulation.
  //

  double totalReactionCount = 0;
  std::size_t numberOfTrajectories = 0;
  ads::Timer timer;
  double elapsedTime = 0;
  // Loop until there are no more tasks.
  while (true) {
    // The number of trajectories to generate in this task.
    std::size_t trajectoriesInTask = 0;
    std::cin >> trajectoriesInTask;
    numberOfTrajectories += trajectoriesInTask;
    if (trajectoriesInTask == 0) {
      break;
    }

    for (std::size_t n = 0; n != trajectoriesInTask; ++n) {
      timer.tic();
      // Run the simulation.
      solver.initialize(initialPopulations, startTime);
      solver.simulate();
      elapsedTime += timer.toc();
      totalReactionCount += solver.getState().getReactionCount();
    }
    // Write the number of trajectories in this task to indicate that the
    // simulations have completed.
    std::cout << trajectoriesInTask << '\n';
    std::cout.flush();
  }
  // Synchronize the histograms.
  solver.synchronize();

  // Empty line for the dictionary of information.
  std::cout << '\n';

  if (! solver.getError().empty()) {
    std::cout << solver.getError() << '\n';
  }
  else {
    // No errors.
    std::cout << '\n';
    // The number of trajectories generated.
    std::cout << numberOfTrajectories << '\n';
    // Write the histograms.
    std::cout << solver.getHistograms();
  }
  // Write the final Mersenne twister state.
  std::cout << solver.getDiscreteUniformGenerator() << '\n';

  if (arePrintingPerformance) {
    // Restore the default precision.
    std::cout.precision(defaultPrecision);
    // Performance message.
    std::cout << "Done.  Simulation time = " << elapsedTime << "\n"
	      << "The ensemble of simulations took " 
	      << totalReactionCount << " steps.\n"
	      << "Reactions per second = " 
	      << totalReactionCount / elapsedTime << ".\n"
	      << "Time per reaction = " 
	      << elapsedTime / totalReactionCount * 1e9 << " nanoseconds\n";
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(0);
    std::cout << elapsedTime / totalReactionCount * 1e9  << "\n";
  }

  return 0;
}

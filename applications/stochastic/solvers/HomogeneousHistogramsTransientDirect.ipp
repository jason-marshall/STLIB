// -*- C++ -*-

#ifndef __HomogeneousHistogramsTransientDirect_ipp__
#error This file is an implementation detail.
#endif

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
      << "stdin and writes the trajectories to stdout.\n";
    // CONTINUE
    exit(1);
  }

}


//! The main loop.
int
main(int argc, char* argv[]) {
  typedef ExponentialGenerator::DiscreteUniformGenerator 
    DiscreteUniformGenerator;
  typedef stochastic::State State;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE
#ifdef STOCHASTIC_SOLVER_PARAMETER
  assert(solverParameters.size() == 1);
#else
  assert(solverParameters.size() == 0);
#endif

  // Number of frames and frame times.
  std::vector<double> frameTimes;
  std::cin >> frameTimes;

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
#ifdef STOCHASTIC_SOLVER_PARAMETER
  Solver solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()), 
		PropensitiesFunctor(reactions), reactionInfluence,
		frameTimes, recordedSpecies, numberOfBins,
		histogramMultiplicity, maximumAllowedSteps,
		solverParameters[0]);
#else
  Solver solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()), 
		PropensitiesFunctor(reactions), reactionInfluence,
		frameTimes, recordedSpecies, numberOfBins,
		histogramMultiplicity, maximumAllowedSteps);
#endif

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
    std::cout << "Simulation time = " << elapsedTime << "\n"
	      << "The ensemble of simulations took " 
	      << totalReactionCount << " steps.\n"
	      << "Reactions per second = " 
	      << totalReactionCount / elapsedTime << ".\n"
	      << "Time per reaction = " 
	      << elapsedTime / totalReactionCount * 1e9 << " nanoseconds\n"
	      << elapsedTime << "\n";
  }

  return 0;
}

// -*- C++ -*-

#ifndef __HomogeneousTauLeapingImplicit_ipp__
#error This file is an implementation detail.
#endif

#include "stochastic/TauLeapingImplicit.h"

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
  typedef stochastic::TauLeapingImplicit Solver;
  typedef stochastic::State State;
  typedef Solver::PropensitiesFunctor PropensitiesFunctor;

  typedef PropensitiesFunctor::ReactionSetType ReactionSet;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE
  assert(solverParameters.size() == 1);

  // Number of frames and frame times.
  std::vector<double> frameTimes;
  std::cin >> frameTimes;

  //
  // Construct the simulation class.
  //
  Solver solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()),
		PropensitiesFunctor(reactions),	maximumAllowedSteps);

  //
  // Read the Mersenne twister state.
  //
  std::cin >> solver.getDiscreteUniformGenerator();

  //
  // Run the simulation.
  //

  // Empty line for the dictionary of information.
  std::cout << '\n';

  // The containers for the populations and the reaction counts.
  std::vector<double> populations(frameTimes.size() * numberOfSpecies);
  std::vector<double> reactionCounts(frameTimes.size() * numberOfReactions);

  double totalReactionCount = 0;
  double totalSteps = 0;
  // Loop until there are no more tasks.
  while (true) {
    // The number of trajectories to generate in this task.
    std::size_t numberOfTrajectories = 0;
    std::cin >> numberOfTrajectories;
    if (numberOfTrajectories == 0) {
      break;
    }
    std::cout << numberOfTrajectories << '\n';

    for (std::size_t n = 0; n != numberOfTrajectories; ++n) {
      // Clear the containers for holding the populations and reaction counts.
      populations.clear();
      reactionCounts.clear();
      // Write the initial Mersenne twister state.
      std::cout << solver.getDiscreteUniformGenerator() << '\n';

      solver.initialize(initialPopulations, startTime);
      for (std::size_t i = 0; i != frameTimes.size(); ++i) {
	// Advance to the next frame.
#ifdef EULER
	solver.simulateFixedEuler(solverParameters[0], frameTimes[i]);
#else
#error "Undefined method."
#endif
	if (! solver.getError().empty()) {
	  break;
	}
	// Record the populations.
	for (std::size_t i = 0; i != recordedSpecies.size(); ++i) {
	  populations.push_back(solver.getState().getPopulations()
				[recordedSpecies[i]]);
	}
	// Record the reaction counts.
	for (std::size_t i = 0; i != recordedReactions.size(); ++i) {
	  reactionCounts.push_back(solver.getState().getReactionCounts()
				   [recordedReactions[i]]);
	}
      }
      totalReactionCount += solver.getState().getReactionCount();
      totalSteps += solver.getStepCount();
      if (! solver.getError().empty()) {
	std::cout << solver.getError() << '\n';
      }
      else {
	std::cout << '\n';
	// Write the populations.
	std::copy(populations.begin(), populations.end(),
		  std::ostream_iterator<double>(std::cout, " "));
	std::cout << '\n';
	// Write the reaction counts.
	std::copy(reactionCounts.begin(), reactionCounts.end(),
		  std::ostream_iterator<double>(std::cout, " "));
	std::cout << '\n';
      }
    }
    // Write the final Mersenne twister state.
    std::cout << solver.getDiscreteUniformGenerator() << '\n';
    std::cout.flush();
  }

  if (arePrintingPerformance) {
    // Restore the default precision.
    std::cout.precision(defaultPrecision);
    // Performance message.
    std::cout << "Reaction count: " << totalReactionCount << '\n'
	      << "Step count: " << totalSteps << '\n';
  }

  return 0;
}

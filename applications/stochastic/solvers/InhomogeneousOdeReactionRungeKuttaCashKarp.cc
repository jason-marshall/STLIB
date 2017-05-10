// -*- C++ -*-

#include "stochastic/OdeReaction.h"
#include "stochastic/PropensitiesInhomogeneous.h"

#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <iterator>
#include <limits>

#include <cassert>

// Problem-specific propensities.
#include "computePropensities.h"

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
  typedef stochastic::OdeReaction<true/*IsInhomogeneous*/,
    stochastic::PropensitiesInhomogeneous<false/*IsDiscrete*/> > Solver;
  typedef Solver::PropensitiesFunctor PropensitiesFunctor;
  typedef PropensitiesFunctor::ReactionSet ReactionSet;
  typedef stochastic::State State;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE: Error message.
  assert(solverParameters.size() == 1);

  // Number of frames and frame times.
  std::vector<double> frameTimes;
  std::cin >> frameTimes;

  //
  // Construct the simulation class.
  //
  Solver solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()),
		PropensitiesFunctor(reactions), maximumAllowedSteps);

  //
  // Run the simulation.
  //

  solver.setupRungeKuttaCashKarp();

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
      populations.clear();
      reactionCounts.clear();
      // Blank line for the MT 19937 state.
      std::cout << '\n';

      bool success = true;
      solver.initialize(initialPopulations, startTime);
      for (std::size_t i = 0; i != frameTimes.size(); ++i) {
	// Advance to the next frame.
	success = solver.simulateRungeKuttaCashKarp(solverParameters[0],
						    frameTimes[i]);
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
	if (! success) {
	  break;
	}
      }
      totalReactionCount += solver.getState().getReactionCount();
      totalSteps += solver.getStepCount();
      if (success) {
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
      else {
	// Error message.
	std::cout << solver.getError();
	const std::vector<double>& populations = 
	  solver.getState().getPopulations();
	for (std::size_t n = 0; n != populations.size(); ++n) {
	  if (populations[n] < 0) {
	    std::cout << " Species " << n + 1
		      << " has the negative population " << populations[n]
		      << ".";
	  }
	}
	std::cout << '\n';
      }
    }
    // Blank line for the final MT 19937 state.
    std::cout << '\n';
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

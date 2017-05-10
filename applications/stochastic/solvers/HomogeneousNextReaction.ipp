// -*- C++ -*-

#ifndef __HomogeneousNextReaction_ipp__
#error This file is an implementation detail.
#endif

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
#include "Propensities.h"
#endif

#include "stochastic/NextReaction.h"
#include "stochastic/ReactionPriorityQueue.h"
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
      << "stdin and writes the trajectories to stdout.\n";
    // CONTINUE
    exit(1);
  }

}


//! The main loop.
int
main(int argc, char* argv[]) {
  typedef stochastic::ReactionPriorityQueue<IndexedPriorityQueue>
    ReactionPriorityQueue;
  typedef ReactionPriorityQueue::DiscreteUniformGenerator
    DiscreteUniformGenerator;

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
  typedef stochastic::NextReaction<ReactionPriorityQueue,
    Propensities<true> > NextReaction;
#else
  typedef stochastic::NextReaction<ReactionPriorityQueue> NextReaction;
#endif

  typedef stochastic::State State;
  typedef NextReaction::PropensitiesFunctor PropensitiesFunctor;
  typedef PropensitiesFunctor::ReactionSetType ReactionSet;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE
  assert(solverParameters.size() == 0);

  // Number of frames and frame times.
  std::vector<double> frameTimes;
  std::cin >> frameTimes;

  //
  // Build the array of reaction influences.
  //
  
  array::StaticArrayOfArrays<std::size_t> reactionInfluence;
  stochastic::computeReactionPropensityInfluence
    (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
     &reactionInfluence, false);

  //
  // Read the Mersenne twister state.
  //
  DiscreteUniformGenerator discreteUniformGenerator;
  std::cin >> discreteUniformGenerator;

  //
  // The reaction priority queue.
  //

#ifdef HASHING
  //std::size_t tableSize = std::max(std::size_t(256), 2 * numberOfReactions);
  std::size_t tableSize = std::min(std::size_t(2048), 4 * numberOfReactions);
  parser.getOption("table", &tableSize);
  if (tableSize < 1) {
    std::cerr << "Bad size for the hash table = " << tableSize << "\n";
    exitOnError();
  }

  //double targetLoad = 2;
  double targetLoad = 8;
  parser.getOption("load", &targetLoad);
  if (targetLoad <= 0) {
    std::cerr << "Bad target load for the hash table = " << targetLoad << "\n";
    exitOnError();
  }
#endif

#ifdef HASHING
  ReactionPriorityQueue reactionPriorityQueue(numberOfReactions,
					      &discreteUniformGenerator,
					      tableSize, targetLoad);
#else
  ReactionPriorityQueue reactionPriorityQueue(numberOfReactions,
					      &discreteUniformGenerator);
#endif

#ifdef COST_CONSTANT
  {
    // The const constant for partitioning methods.
    double cost;
    if (parser.getOption("cost", &cost)) {
      reactionPriorityQueue.setCostConstant(cost);
    }
  }
#endif

  // There should be no more options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  //
  // Construct the simulation class.
  //
  NextReaction solver(State(initialPopulations, reactions.getBeginning(),
			    reactions.getEnd()),
		      PropensitiesFunctor(reactions), &reactionPriorityQueue,
		      reactionInfluence, maximumAllowedSteps);

  //
  // Run the simulation.
  //

  // Empty line for the dictionary of information.
  std::cout << '\n';

  // The containers for the populations and the reaction counts.
  std::vector<double> populations(frameTimes.size() * numberOfSpecies);
  std::vector<double> reactionCounts(frameTimes.size() * numberOfReactions);

  double totalReactionCount = 0;
  ads::Timer timer;
  double elapsedTime = 0;
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

      timer.tic();
      solver.initialize(initialPopulations, startTime);
      for (std::size_t i = 0; i != frameTimes.size(); ++i) {
	// Advance to the next frame.
	solver.simulate(frameTimes[i]);
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
      elapsedTime += timer.toc();
      totalReactionCount += solver.getState().getReactionCount();
      if (! solver.getError().empty()) {
	std::cout << solver.getError() << '\n';
      }
      else {
	// No errors.
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
    std::cout << "Done. Meaningless result = " 
	      << discreteUniformGenerator() <<'\n'
	      << "Simulation time = " << elapsedTime << "\n"
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

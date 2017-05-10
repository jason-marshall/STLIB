// -*- C++ -*-

#ifndef __HomogeneousDirectAllReactions_ipp__
#error This file is an implementation detail.
#endif

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
#include "Propensities.h"
#endif

#include "stochastic/Direct.h"
#include "stochastic/ReactionSet.h"
#include "stochastic/reactionPropensityInfluence.h"

#include "ads/timer/Timer.h"
#include "ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <iterator>

#include <cassert>

#define STOCHASTIC_USE_INFLUENCE

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

#ifdef STOCHASTIC_CUSTOM_PROPENSITIES
  typedef Propensities<true> PropensitiesFunctor;
#else
#ifdef STOCHASTIC_USE_INFLUENCE
  // If we use the reaction influence array, we will compute the propensities
  // one at a time.
  typedef stochastic::PropensitiesSingle<true> PropensitiesFunctor;
#else
  // Otherwise, we compute the propensities all at once.
  typedef stochastic::PropensitiesAll<true> PropensitiesFunctor;
#endif
#endif

  typedef PropensitiesFunctor::ReactionSetType ReactionSet;
  typedef stochastic::Direct<DiscreteGenerator, ExponentialGenerator,
    PropensitiesFunctor> Direct;

  typedef stochastic::State State;

#define __input_ipp__
#include "input.ipp"
#undef __input_ipp__

  // Check the number of solver parameters.
  // CONTINUE
  assert(solverParameters.size() == 0);

  // Equilibration time.
  double equilibrationTime;
  std::cin >> equilibrationTime;

  // Recording time.
  double recordingTime;
  std::cin >> recordingTime;

  //
  // Build the array of reaction influences.
  //
  
  // If not used, this is left empty.
  array::StaticArrayOfArrays<std::size_t> reactionInfluence;
#ifdef STOCHASTIC_USE_INFLUENCE
  stochastic::computeReactionPropensityInfluence
    (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
     &reactionInfluence, true);
#endif

  //
  // Construct the simulation class.
  //
  Direct solver(State(initialPopulations, reactions.getBeginning(),
		      reactions.getEnd()), 
		PropensitiesFunctor(reactions), reactionInfluence,
		maximumAllowedSteps);

  //
  // Read the Mersenne twister state.
  //
  std::cin >> solver.getDiscreteUniformGenerator();

#ifdef STOCHASTIC_USE_INFLUENCE_IN_GENERATOR
  solver.getDiscreteGenerator().setInfluence(&reactionInfluence);
#endif

  // CONTINUE
  //
  // Get more repairing and rebuilding options.
  //

#ifdef STOCHASTIC_REPAIR
  // For loosely coupled problems, you can specify how many steps to take
  // between repairs.
  DiscreteGenerator::Counter stepsBetweenRepairs = 0;
  if (parser.getOption("repair", &stepsBetweenRepairs)) {
    if (stepsBetweenRepairs < 1) {
      std::cerr
	<< "Error: Bad value for the number of steps between repairs.\n";
      exitOnError();
    }
    solver.getDiscreteGenerator().
      setStepsBetweenRepairs(stepsBetweenRepairs);
  }
#endif

#ifdef STOCHASTIC_REBUILD
  // The number of steps to take between rebuilds.
  DiscreteGenerator::Counter stepsBetweenRebuilds = 0;
  if (parser.getOption("rebuild", &stepsBetweenRebuilds)) {
    if (stepsBetweenRebuilds < 1) {
      std::cerr
	<< "Error: Bad value for the number of steps between rebuilds.\n";
      exitOnError();
    }
    solver.getDiscreteGenerator().
      setStepsBetweenRebuilds(stepsBetweenRebuilds);
  }
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

  // Empty line for the dictionary of information.
  std::cout << '\n';

  // The containers for the reaction indices and times.
  std::vector<double> equilibratedPopulations(numberOfSpecies);
  std::vector<std::size_t> reactionIndices;
  std::back_insert_iterator<std::vector<std::size_t> >
    indicesIterator(reactionIndices);
  std::vector<double> reactionTimes;
  std::back_insert_iterator<std::vector<double> > timesIterator(reactionTimes);
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
      // Clear the containers for holding the reaction indices and times.
      reactionIndices.clear();
      reactionTimes.clear();
      // Write the initial Mersenne twister state.
      std::cout << solver.getDiscreteUniformGenerator() << '\n';

      timer.tic();
      solver.initialize(initialPopulations, startTime);
      // Equilibrate
      if (equilibrationTime != 0) {
	solver.equilibrate(equilibrationTime);
      }
      std::copy(solver.getState().getPopulations().begin(),
		solver.getState().getPopulations().end(),
		equilibratedPopulations.begin());
      // Simulate and record.
      solver.simulate(startTime + equilibrationTime + recordingTime,
		      indicesIterator, timesIterator);
      elapsedTime += timer.toc();
      totalReactionCount += solver.getState().getReactionCount();
      if (! solver.getError().empty()) {
	std::cout << solver.getError() << '\n';
      }
      else {
	// No errors.
	std::cout << '\n';
	// Write the initial (possibly equilibrated) populations.
	std::copy(equilibratedPopulations.begin(),
		  equilibratedPopulations.end(),
		  std::ostream_iterator<double>(std::cout, " "));
	std::cout << '\n';
	// Write the reaction indices.
	std::copy(reactionIndices.begin(), reactionIndices.end(),
		  std::ostream_iterator<std::size_t>(std::cout, " "));
	std::cout << '\n';
	// Write the reaction times.
	std::cout.precision(16);
	std::copy(reactionTimes.begin(), reactionTimes.end(),
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
    std::cout.precision(3);
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

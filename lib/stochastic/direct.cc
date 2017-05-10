// -*- C++ -*-

#include "direct.h"

#include "stochastic/Direct.h"
#include "stochastic/Propensities.h"
#include "stochastic/reactionPropensityInfluence.h"

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGeneratorRejectionBinsSplitting.h"

#include "ads/array/SparseArray.h"

#include <iostream>

unsigned
generateMt19937State(unsigned seed, unsigned state[]) {
  return numerical::DiscreteUniformGeneratorMt19937::generateState(seed, state);
}

int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[],
	 unsigned mt19937state[]) {
  typedef double Number;
  typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
  typedef numerical::DiscreteGeneratorRejectionBinsSplitting<true, true>
    DiscreteGenerator;

  typedef ads::Array<1, int> PopulationArray;
  typedef ads::SparseArray<1, int> StateChangeArray;
  typedef stochastic::State<PopulationArray,
    ads::Array<1, StateChangeArray> > State;

  typedef stochastic::ReactionSet<Number> ReactionSet;
  typedef ReactionSet::Reaction Reaction;

  typedef stochastic::PropensitiesSingle<Number> PropensitiesFunctor;

  const int N = numerical::mt19937mn::N;
  assert(maximumAllowedSteps <= double(std::numeric_limits<long>::max()));
  const std::size_t maxSteps = (maximumAllowedSteps > 0 ? 
				std::size_t(maximumAllowedSteps) :
				std::numeric_limits<std::size_t>::max());

#if 0
  {
    //
    // Debugging code.
    //
    std::cout << "Number of species = " << numberOfSpecies << "\n";
    std::cout << "Initial populations:\n";
    for (int i = 0; i != numberOfSpecies; ++i) {
      std::cout << initialPopulationsArray[i] << "\n";
    }
    std::cout << "Number of reactions = " << numberOfReactions << "\n";
    const int* data = packedReactions;
    for (int i = 0; i != numberOfReactions; ++i) {
      int numberOfReactants = *data++;
      for (int j = 0; j != numberOfReactants; ++j) {
	int index = *data++;
	int stoichiometry = *data++;
	std::cout << stoichiometry << " " << index << "  ";
      }
      std::cout << "->";
      int numberOfProducts = *data++;
      for (int j = 0; j != numberOfProducts; ++j) {
	int index = *data++;
	int stoichiometry = *data++;
	std::cout << "  " << stoichiometry << " " << index;
      }
      std::cout << ", Propensity factor = " << propensityFactors[i] << "\n";
    }
  }
#endif

  // Record the initial populations.
  State::PopulationsContainer initialPopulations(numberOfSpecies,
						 initialPopulationsArray);

  // Construct the reaction set.
  std::vector<Reaction> reactionsVector(numberOfReactions);
  {
    std::vector<int> reactantIndices, reactantStoichiometries,
      productIndices, productStoichiometries;
    const int* data = packedReactions;
    for (int i = 0; i != numberOfReactions; ++i) {
      reactantIndices.clear();
      reactantStoichiometries.clear();
      productIndices.clear();
      productStoichiometries.clear();

      int numberOfReactants = *data++;
      for (int j = 0; j != numberOfReactants; ++j) {
	reactantIndices.push_back(*data++);
	reactantStoichiometries.push_back(*data++);
      }
      int numberOfProducts = *data++;
      for (int j = 0; j != numberOfProducts; ++j) {
	productIndices.push_back(*data++);
	productStoichiometries.push_back(*data++);
      }
      reactionsVector[i] = 
	Reaction(reactantIndices.begin(), reactantIndices.end(),
		 reactantStoichiometries.begin(), reactantStoichiometries.end(),
		 productIndices.begin(), productIndices.end(),
		 productStoichiometries.begin(), productStoichiometries.end(),
		 propensityFactors[i]);
    }
  }
  stochastic::ReactionSet<Number> reactions(reactionsVector.begin(),
					    reactionsVector.end());
  
  //
  // Build the state change vectors.
  //

  State::ScvContainer stateChangeVectors;
  stochastic::buildStateChangeVectors
    (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
     &stateChangeVectors);

  //
  // Build the array of reaction influences.
  //
  
  ads::StaticArrayOfArrays<int> reactionInfluence;
  stochastic::computeReactionPropensityInfluence
    (initialPopulations.size(), reactions.getBeginning(), reactions.getEnd(),
     &reactionInfluence, true);

  //
  // The propensities functor.
  //

  PropensitiesFunctor propensitiesFunctor(reactions);


  // Construct the state class.
  State state(initialPopulations, stateChangeVectors);

  // Check the validity of the initial state.
  if (! state.isValid()) {
    std::cerr << "Error: The initial state of the simulation is not valid.\n";
    return false;
  }

  // Construct the simulation class.
  stochastic::Direct<State, PropensitiesFunctor, ExponentialGenerator,
    DiscreteGenerator> direct(state, propensitiesFunctor, 
				    reactionInfluence);

  // If a state vector was specified, instead of a null pointer.
  if (mt19937state) {
    // Set the state of the Mersenne twister.
    for (int i = 0; i != N; ++i) {
      direct.getDiscreteUniformGenerator().setState(i, 
						    unsigned(mt19937state[i]));
    }
  }

  // Run the simulation.
  direct.initialize(initialPopulations, startTime);
  int* populationIterator = framePopulations;
  std::size_t* reactionCountIterator = frameReactionCounts;
  for (int i = 0; i != numberOfFrames; ++i) {
    // Make a termination condition.
    stochastic::EssTerminationConditionEndTimeReactionCount<Number>
      terminationCondition(frameTimes[i], maxSteps);
    // Simulate up to the termination condition.
    direct.simulate(terminationCondition);
    // Record the populations and reaction counts.
    for (int species = 0; species != numberOfSpecies; ++species) {
      *populationIterator++ = direct.getState().getPopulation(species);
    }
    for (int reaction = 0; reaction != numberOfReactions; ++reaction) {
      *reactionCountIterator++ = direct.getState().getReactionCount(reaction);
    }
  }

  // If a state vector was specified, instead of a null pointer.
  if (mt19937state) {
    // Get the new state of the Mersenne twister.
    for (int i = 0; i != N; ++i) {
      mt19937state[i] = direct.getDiscreteUniformGenerator().getState(i);
    }
  }

  return true;
}

int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[]) {
  return simulate(numberOfSpecies, initialPopulationsArray,
		  numberOfReactions, packedReactions, 
		  propensityFactors, startTime, maximumAllowedSteps,
		  numberOfFrames, frameTimes,
		  framePopulations, frameReactionCounts, 0);
}

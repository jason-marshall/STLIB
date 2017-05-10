// -*- C++ -*-

#include <cstddef>

unsigned
generateMt19937State(unsigned seed, unsigned state[]);

int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[],
	 unsigned mt19937state[]);

int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[]);

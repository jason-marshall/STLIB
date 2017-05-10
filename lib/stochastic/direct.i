// -*- C++ -*-

%module direct
%include "carrays.i"
%array_class(int, ArrayInt);
%array_class(unsigned, ArrayUnsigned);
%array_class(std::size_t, ArraySizeType);
%array_class(double, ArrayDouble);

%{
#include "direct.h"
%}

extern
unsigned
generateMt19937State(unsigned seed, unsigned state[]);

extern
int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[],
	 unsigned mt19937state[]);

extern
int
simulate(int numberOfSpecies, int initialPopulationsArray[],
	 int numberOfReactions, int packedReactions[], 
	 double propensityFactors[],
	 double startTime, std::size_t maximumAllowedSteps,
	 int numberOfFrames, double frameTimes[],
	 int framePopulations[], std::size_t frameReactionCounts[]);

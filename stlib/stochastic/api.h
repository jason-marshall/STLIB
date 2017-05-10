// -*- C++ -*-

#if !defined(__stochastic_api_h__)
#define __stochastic_api_h__

// CONTINUE: I used this for Py_BEGIN_ALLOW_THREADS
//#include "Python.h"

#include "stlib/stochastic/Direct.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/reactionPropensityInfluence.h"

// CONTINUE REMOVE
#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplitting.h"

#include <iostream>

namespace stlib
{
namespace stochastic
{

//! Build a new solver that uses the direct method.
template<typename _Solver>
_Solver*
newSolverDirect(std::size_t numberOfSpecies,
                std::size_t numberOfReactions,
                const std::size_t packedReactions[],
                const double propensityFactors[]);

//! Delete the solver.
template<typename _Solver>
void
deleteSolver(_Solver* solver);

//! Generate the state vector for the Mersenne Twister from the seed.
/*! \return A new seed. */
unsigned
generateMt19937State(unsigned seed, unsigned state[]);

//! Get the state of the Mersenne twister.
template<typename _Solver>
void
getMt19937State(const _Solver* solver, unsigned state[]);

//! Set the state of the Mersenne twister.
template<typename _Solver>
void
setMt19937State(_Solver* solver, const unsigned state[]);

//! Generate a trajectory.
template<typename _Solver>
int
generateTrajectory(_Solver* solver, const std::size_t initialPopulationsArray[],
                   double startTime, std::size_t maximumAllowedSteps,
                   std::size_t numberOfFrames, const double frameTimes[],
                   std::size_t framePopulations[],
                   std::size_t frameReactionCounts[]);

//! Generate a trajectory.
int
simulate(std::size_t numberOfSpecies, std::size_t initialPopulationsArray[],
         std::size_t numberOfReactions, std::size_t packedReactions[],
         double propensityFactors[],
         double startTime, std::size_t maximumAllowedSteps,
         std::size_t numberOfFrames, double frameTimes[],
         std::size_t framePopulations[], std::size_t frameReactionCounts[],
         unsigned mt19937state[]);

//! Generate a trajectory.
int
simulate(std::size_t numberOfSpecies, std::size_t initialPopulationsArray[],
         std::size_t numberOfReactions, std::size_t packedReactions[],
         double propensityFactors[],
         double startTime, std::size_t maximumAllowedSteps,
         std::size_t numberOfFrames, double frameTimes[],
         std::size_t framePopulations[], std::size_t frameReactionCounts[]);

} // namespace stochastic
}

#define __stochastic_api_ipp__
#include "stlib/stochastic/api.ipp"
#undef __stochastic_api_ipp__

#endif

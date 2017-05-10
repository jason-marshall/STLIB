// -*- C++ -*-

#if !defined(__stochastic_api_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace stochastic
{

// Build a new solver that uses the direct method.
template<typename _Solver>
inline
_Solver*
newSolverDirect(const std::size_t numberOfSpecies,
                const std::size_t numberOfReactions,
                const std::size_t packedReactions[],
                const double propensityFactors[])
{
  typedef typename _Solver::PropensitiesFunctor PropensitiesFunctor;
  typedef typename _Solver::ExponentialGenerator ExponentialGenerator;
  typedef typename _Solver::DiscreteGenerator DiscreteGenerator;
  typedef ReactionSet<true> ReactionSet;
  typedef typename ReactionSet::ReactionType Reaction;

  // Construct the reaction set.
  std::vector<Reaction> reactionsVector(numberOfReactions);
  {
    std::vector<std::pair<std::size_t, std::size_t> >& reactants, products;
    const std::size_t* data = packedReactions;
    for (std::size_t i = 0; i != numberOfReactions; ++i) {
      reactants.clear();
      products.clear();

      std::size_t numberOfReactants = *data++;
      for (std::size_t j = 0; j != numberOfReactants; ++j) {
        reactants.push_back(std::make_pair(data[0], data[1]));
        data += 2;
      }
      std::size_t numberOfProducts = *data++;
      for (std::size_t j = 0; j != numberOfProducts; ++j) {
        products.push_back(std::make_pair(data[0], data[1]));
        data += 2;
      }
      reactionsVector[i] =
        Reaction(reactants, products,
                 propensityFactors[i]);
    }
  }
  ReactionSet reactions(reactionsVector.begin(), reactionsVector.end());

  //
  // Build the state change vectors.
  //

  typename State::ScvContainer stateChangeVectors;
  buildStateChangeVectors(numberOfSpecies, reactions.getBeginning(),
                          reactions.getEnd(), &stateChangeVectors);

  //
  // Build the array of reaction influences.
  //

  container::StaticArrayOfArrays<std::size_t> reactionInfluence;
  computeReactionPropensityInfluence
  (numberOfSpecies, reactions.getBeginning(), reactions.getEnd(),
   &reactionInfluence, true);

  // The propensities functor.
  PropensitiesFunctor propensitiesFunctor(reactions);

  // Construct the state class.
  State state(numberOfSpecies, stateChangeVectors);

  // Check the validity of the initial state.
  assert(state.isValid());

  // Construct the simulation class.
  return new _Solver(state, propensitiesFunctor, reactionInfluence);
}


// Delete the solver.
template<typename _Solver>
inline
void
deleteSolver(_Solver* solver)
{
  assert(solver);
  delete solver;
  solver = 0;
}

// Generate the state vector for the Mersenne Twister from the seed.
// Return a new seed.
inline
unsigned
generateMt19937State(const unsigned seed, unsigned state[])
{
  return numerical::DiscreteUniformGeneratorMt19937::generateState(seed, state);
}

// Get the state of the Mersenne twister.
template<typename _Solver>
inline
void
getMt19937State(const _Solver* const solver, unsigned state[])
{
  solver->getDiscreteUniformGenerator().getState(state);
}

// Set the state of the Mersenne twister.
template<typename _Solver>
inline
void
setMt19937State(_Solver* const solver, const unsigned state[])
{
  solver->getDiscreteUniformGenerator().setState(state);
}

// CONTINUE: Think about the return value.
// Generate a trajectory.
template<typename _Solver>
inline
int
generateTrajectory(_Solver* const solver,
                   const std::size_t initialPopulationsArray[],
                   const double startTime, std::size_t maximumAllowedSteps,
                   const std::size_t numberOfFrames, const double frameTimes[],
                   std::size_t framePopulations[],
                   std::size_t frameReactionCounts[])
{

  // CONTINUE
  //Py_BEGIN_ALLOW_THREADS

  const std::size_t numberOfSpecies = solver->getState().getNumberOfSpecies();
  const std::size_t numberOfReactions =
    solver->getState().getNumberOfReactions();

  // If they did not specify a maximum allowed number of steps.
  if (maximumAllowedSteps == 0) {
    maximumAllowedSteps = std::numeric_limits<std::size_t>::max();
  }

  // Copy the initial populations.
  typename std::vector<double>
  initialPopulations(numberOfSpecies);
  std::copy(initialPopulationsArray, initialPopulationsArray + numberOfSpecies,
            initialPopulations.begin());
  // Set the initial population and the starting time.
  solver->initialize(initialPopulations, startTime);

  // Run the simulation.
  std::size_t* populationIterator = framePopulations;
  std::size_t* reactionCountIterator = frameReactionCounts;
  for (std::size_t i = 0; i != numberOfFrames; ++i) {
    // Make a termination condition.
    EssTerminationConditionEndTimeReactionCount<double>
    terminationCondition(frameTimes[i], maximumAllowedSteps);
    // Simulate up to the termination condition.
    solver->simulate(terminationCondition);
    // Record the populations and reaction counts.
    for (std::size_t species = 0; species != numberOfSpecies; ++species) {
      *populationIterator++ = solver->getState().getPopulation(species);
    }
    for (std::size_t reaction = 0; reaction != numberOfReactions; ++reaction) {
      *reactionCountIterator++ = solver->getState().getReactionCount(reaction);
    }
  }

  // CONTINUE
  //Py_END_ALLOW_THREADS

  return true;
}


inline
int
simulate(std::size_t numberOfSpecies, std::size_t initialPopulationsArray[],
         std::size_t numberOfReactions, std::size_t packedReactions[],
         double propensityFactors[],
         double startTime, std::size_t maximumAllowedSteps,
         std::size_t numberOfFrames, double frameTimes[],
         std::size_t framePopulations[], std::size_t frameReactionCounts[],
         unsigned mt19937state[])
{
  typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
  typedef numerical::DiscreteGeneratorRejectionBinsSplitting<true, true>
  DiscreteGenerator;

  typedef ReactionSet<double> ReactionSet;
  typedef ReactionSet::Reaction Reaction;

  typedef PropensitiesSingle<double> PropensitiesFunctor;

  const std::size_t N = numerical::mt19937mn::N;
  assert(maximumAllowedSteps <= double(std::numeric_limits<long>::max()));
  const std::size_t maxSteps = (maximumAllowedSteps > 0 ?
                                std::size_t(maximumAllowedSteps) :
                                std::numeric_limits<std::size_t>::max());

  // Record the initial populations.
  std::vector<double> initialPopulations(numberOfSpecies);
  std::copy(initialPopulationsArray, initialPopulationsArray + numberOfSpecies,
            initialPopulations.begin());

  // Construct the reaction set.
  std::vector<Reaction> reactionsVector(numberOfReactions);
  {
    std::vector<std::size_t> reactantIndices, reactantStoichiometries,
        productIndices, productStoichiometries;
    const std::size_t* data = packedReactions;
    for (std::size_t i = 0; i != numberOfReactions; ++i) {
      reactantIndices.clear();
      reactantStoichiometries.clear();
      productIndices.clear();
      productStoichiometries.clear();

      std::size_t numberOfReactants = *data++;
      for (std::size_t j = 0; j != numberOfReactants; ++j) {
        reactantIndices.push_back(*data++);
        reactantStoichiometries.push_back(*data++);
      }
      std::size_t numberOfProducts = *data++;
      for (std::size_t j = 0; j != numberOfProducts; ++j) {
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
  ReactionSet reactions(reactionsVector.begin(), reactionsVector.end());

  //
  // Build the state change vectors.
  //

  State::ScvContainer stateChangeVectors;
  buildStateChangeVectors(initialPopulations.size(), reactions.getBeginning(),
                          reactions.getEnd(), &stateChangeVectors);

  //
  // Build the array of reaction influences.
  //

  container::StaticArrayOfArrays<std::size_t> reactionInfluence;
  computeReactionPropensityInfluence
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
  Direct<DiscreteGenerator, ExponentialGenerator, PropensitiesFunctor>
  direct(state, propensitiesFunctor, reactionInfluence);

  // If a state vector was specified, instead of a null pointer.
  if (mt19937state) {
    // Set the state of the Mersenne twister.
    direct.getDiscreteUniformGenerator().setState(mt19937state);
  }

  // Run the simulation.
  direct.initialize(initialPopulations, startTime);
  std::size_t* populationIterator = framePopulations;
  std::size_t* reactionCountIterator = frameReactionCounts;
  for (std::size_t i = 0; i != numberOfFrames; ++i) {
    // Make a termination condition.
    EssTerminationConditionEndTimeReactionCount<double>
    terminationCondition(frameTimes[i], maxSteps);
    // Simulate up to the termination condition.
    direct.simulate(terminationCondition);
    // Record the populations and reaction counts.
    for (std::size_t species = 0; species != numberOfSpecies; ++species) {
      *populationIterator++ = direct.getState().getPopulation(species);
    }
    for (std::size_t reaction = 0; reaction != numberOfReactions; ++reaction) {
      *reactionCountIterator++ = direct.getState().getReactionCount(reaction);
    }
  }

  // If a state vector was specified, instead of a null pointer.
  if (mt19937state) {
    // Get the new state of the Mersenne twister.
    for (std::size_t i = 0; i != N; ++i) {
      mt19937state[i] = direct.getDiscreteUniformGenerator().getState(i);
    }
  }

  return true;
}

inline
int
simulate(std::size_t numberOfSpecies, std::size_t initialPopulationsArray[],
         std::size_t numberOfReactions, std::size_t packedReactions[],
         double propensityFactors[],
         double startTime, std::size_t maximumAllowedSteps,
         std::size_t numberOfFrames, double frameTimes[],
         std::size_t framePopulations[], std::size_t frameReactionCounts[])
{
  return simulate(numberOfSpecies, initialPopulationsArray,
                  numberOfReactions, packedReactions,
                  propensityFactors, startTime, maximumAllowedSteps,
                  numberOfFrames, frameTimes,
                  framePopulations, frameReactionCounts, 0);
}

} // namespace stochastic
}

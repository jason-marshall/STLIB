// -*- C++ -*-

// CONTINUE
#if 0
#include "stlib/stochastic/TauLeapingData.h"
#include "stlib/stochastic/ReactionSet.h"
#endif

int
main()
{
  // CONTINUE
#if 0
  typedef stochastic::TauLeapingData<double> TauLeapingData;
  typedef TauLeapingData::State State;
  typedef stochastic::Reaction<> Reaction;
  typedef stochastic::ReactionSet<> ReactionSet;

  //-------------------------------------------------------------------------
  // Empty state.
  {
    const std::size_t NumberOfSpecies = 0;
    const double Epsilon = 0.1;
    TauLeapingData x(NumberOfSpecies, Epsilon);

    assert(x.getEpsilon() == Epsilon);
    x.setEpsilon(1);
    assert(x.getEpsilon() == 1);

    ReactionSet emptyReactionSet;
    State emptyState(&emptyReactionSet, 0);
    x.initialize(emptyState);
    assert(x.computeTau(emptyState) == std::numeric_limits<double>::max());
  }

  //-------------------------------------------------------------------------
  // s0 -> s1
  {
    // Construct the reaction.
    std::vector<std::size_t> reactantIndices, reactantCoefficients,
        productIndices, productCoefficients;
    reactantIndices.push_back(0);
    reactantCoefficients.push_back(1);
    productIndices.push_back(1);
    productCoefficients.push_back(1);
    const double RateConstant = 2.0;
    Reaction reaction(reactantIndices.begin(), reactantIndices.end(),
                      reactantCoefficients.begin(), reactantCoefficients.end(),
                      productIndices.begin(), productIndices.end(),
                      productCoefficients.begin(), productCoefficients.end(),
                      RateConstant);

    // Construct the state.
    ReactionSet reactionSet(1);
    reactionSet.insertReaction(reaction);
    State state(&reactionSet, 2);
    state.setPopulation(0, 0);
    state.setPopulation(1, 0);
    state.computePropensityFunction(0);

    const double Epsilon = 0.1;
    TauLeapingData x(state.getNumberOfSpecies(), Epsilon);

    assert(x.getEpsilon() == Epsilon);
    x.setEpsilon(1);
    assert(x.getEpsilon() == 1);

    x.initialize(state);
    assert(x.computeTau(state) == std::numeric_limits<double>::max());
    state.setPopulation(0, 1);
    state.setPopulation(1, 1);
    state.computePropensityFunction(0);
    assert(std::abs(x.computeTau(state) - 0.5) <
           10 * std::numeric_limits<double>::epsilon());
  }
#endif
  return 0;
}

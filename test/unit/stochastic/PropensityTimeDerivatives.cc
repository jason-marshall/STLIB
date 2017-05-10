// -*- C++ -*-

#include "stlib/stochastic/PropensityTimeDerivatives.h"

using namespace stlib;

int
main()
{
  {
    //
    // The population and reaction count types are double.
    //
    using stochastic::State;
    using stochastic::PropensityTimeDerivatives;
    typedef stochastic::ReactionSet<true> ReactionSet;
    typedef ReactionSet::ReactionType Reaction;

    //-----------------------------------------------------------------------
    // s0 -> s1
    {
      // a = 2 s0
      // ds0/dt = (-1) a = -2 s0
      // ds1/dt = (1) a = 2 s0
      // da/dt = pa/ps0 ds0/dt = 2 (-2 s0) = -4 s0

      // Construct the reaction.
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction reaction(reactants, products, dependencies, RateConstant);

      ReactionSet reactionSet(&reaction, &reaction + 1);

      // Construct from the populations and the reactions.
      State state(2, reactionSet.getBeginning(), reactionSet.getEnd());
      std::vector<double> a(state.getNumberOfReactions());
      std::vector<double> first(state.getNumberOfReactions());

      PropensityTimeDerivatives ptd(state.getNumberOfSpecies());

      state.setPopulation(0, 0);
      state.setPopulation(1, 0);
      a[0] = reactionSet.computePropensity(0, state.getPopulations());
      ptd(state, reactionSet, a, &first);
      assert(first[0] == -4 * state.getPopulation(0));
    }
  }

  return 0;
}

// -*- C++ -*-

#include "stlib/stochastic/State.h"
#include "stlib/stochastic/ReactionSet.h"

USING_STLIB_EXT_PAIR_IO_OPERATORS;
using namespace stlib;

int
main()
{
  {
    //
    // The population and reaction count types are double.
    //
    typedef stochastic::State State;
    typedef stochastic::ReactionSet<true> ReactionSet;
    typedef ReactionSet::ReactionType Reaction;

    //-----------------------------------------------------------------------
    // Empty state.
    {
      std::vector<double> populations;
      ReactionSet reactionSet;
      State x(populations, reactionSet.getBeginning(), reactionSet.getEnd());
      assert(x.getReactionCount() == 0);
      assert(x.getNumberOfSpecies() == 0);
      assert(x.getNumberOfReactions() == 0);
      assert(x.isValid());
    }

    //-----------------------------------------------------------------------
    // s0 -> s1
    {
      // Construct the reaction.
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction reaction(reactants, products, dependencies, RateConstant);

      ReactionSet reactionSet(&reaction, &reaction + 1);

      {
        std::vector<double> populations(2);
        // Construct from the populations and the reactions.
        State x(populations, reactionSet.getBeginning(),
                reactionSet.getEnd());
        assert(x.getReactionCount() == 0);
        assert(x.getNumberOfSpecies() == 2);
        assert(x.getNumberOfReactions() == 1);
        x.setPopulation(0, 0);
        x.setPopulation(1, 0);
        assert(reactionSet.getReaction(0).
               computePropensityFunction(x.getPopulations()) == 0);
        assert(x.isValid());

        // Derivatives.
        std::vector<double> propensities(x.getNumberOfReactions());
        std::vector<double> dxdt(x.getNumberOfSpecies());
        for (std::size_t j = 0; j != propensities.size(); ++j) {
          propensities[j] =
            reactionSet.computePropensity(j, x.getPopulations());
        }
        x.populationDerivatives(propensities, &dxdt);
        assert(dxdt[0] == 0);
        assert(dxdt[1] == 0);

        x.setPopulation(0, 0);
        x.setPopulation(1, 10);
        for (std::size_t j = 0; j != propensities.size(); ++j) {
          propensities[j] =
            reactionSet.computePropensity(j, x.getPopulations());
        }
        x.populationDerivatives(propensities, &dxdt);
        assert(dxdt[0] == 0);
        assert(dxdt[1] == 0);

        x.setPopulation(0, 1);
        x.setPopulation(1, 0);
        for (std::size_t j = 0; j != propensities.size(); ++j) {
          propensities[j] =
            reactionSet.computePropensity(j, x.getPopulations());
        }
        x.populationDerivatives(propensities, &dxdt);
        assert(dxdt[0] == - RateConstant);
        assert(dxdt[1] == RateConstant);

        x.setPopulation(0, 2);
        x.setPopulation(1, 0);
        for (std::size_t j = 0; j != propensities.size(); ++j) {
          propensities[j] =
            reactionSet.computePropensity(j, x.getPopulations());
        }
        x.populationDerivatives(propensities, &dxdt);
        assert(dxdt[0] == - 2 * RateConstant);
        assert(dxdt[1] == 2 * RateConstant);
      }

      {
      }
    }
  }

  return 0;
}

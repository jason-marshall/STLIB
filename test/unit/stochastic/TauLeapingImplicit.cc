// -*- C++ -*-

#include "stlib/stochastic/TauLeapingImplicit.h"

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

int
main()
{
  {
    //
    // The population and reaction count types are double.
    //
    typedef stochastic::TauLeapingImplicit TauLeapingImplicit;
    typedef stochastic::State State;
    typedef TauLeapingImplicit::PropensitiesFunctor PropensitiesFunctor;

    typedef stochastic::ReactionSet<true> ReactionSet;
    typedef ReactionSet::ReactionType Reaction;

    //-------------------------------------------------------------------------
    // s0 -> s1
    // s1 -> s0
    {
      // Construct the reactions.
      std::vector<Reaction> reactions;
      {
        // s0 -> s1
        std::vector<std::pair<std::size_t, std::size_t> > reactants,
            products;
        reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
        products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
        std::vector<std::size_t> dependencies(1, 0);
        const double RateConstant = 1.0;
        reactions.push_back(Reaction(reactants, products, dependencies,
                                     RateConstant));
      }
      {
        // s1 -> s0
        std::vector<std::pair<std::size_t, std::size_t> > reactants,
            products;
        reactants.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
        products.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
        std::vector<std::size_t> dependencies(1, 1);
        const double RateConstant = 2.0;
        reactions.push_back(Reaction(reactants, products, dependencies,
                                     RateConstant));
      }
      ReactionSet reactionSet(reactions.begin(), reactions.end());

      // Initial populations.
      std::vector<double> populations(2);
      populations[0] = 10;
      populations[1] = 0;

      // The state.
      State state(populations.size(), reactionSet.getBeginning(),
                  reactionSet.getEnd());

      // Propensities functor.
      PropensitiesFunctor propensitiesFunctor(reactionSet);

      // The solver.
      TauLeapingImplicit solver(state, propensitiesFunctor, 1e9);
      solver.initialize(populations, 0.);

      // Simulate.
      solver.simulateFixedEuler(1., 1.);
      assert(solver.getError().empty());

      std::cout << "populations = \n" << solver.getState().getPopulations();
    }
  }

  return 0;
}

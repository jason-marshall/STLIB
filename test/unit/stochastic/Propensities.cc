// -*- C++ -*-

#include "stlib/stochastic/Propensities.h"

using namespace stlib;

int
main()
{
  typedef stochastic::PropensitiesSingle<true> Propensities;
  typedef Propensities::ReactionSetType ReactionSet;
  typedef ReactionSet::ReactionType Reaction;

  {
    // Default constructor.
    ReactionSet rs;
    Propensities x(rs);
    assert(x.getSize() == 0);
  }
  {
    //------------------------------------------------------------------------
    // s0 -> s1
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
    products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
    std::vector<std::size_t> dependencies(1, 0);
    const double RateConstant = 2.0;
    Reaction reaction(reactants, products, dependencies, RateConstant);

    Propensities x(ReactionSet(&reaction, &reaction + 1));
    assert(x.getSize() == 1);
    assert(x.getReaction(0) == reaction);
    std::vector<double> populations;
    populations.push_back(3);
    assert(x(0, populations) == RateConstant * populations[0]);
  }

  return 0;
}

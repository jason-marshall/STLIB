// -*- C++ -*-

#include "stlib/stochastic/ReactionSet.h"

using namespace stlib;

int
main()
{
  typedef stochastic::ReactionSet<true> ReactionSet;
  typedef ReactionSet::ReactionType Reaction;

  {
    // Default constructor.
    ReactionSet x;
    assert(x.getSize() == 0);
    assert(x.getBeginning() == x.getEnd());
    // CONTINU REMOVE
    //assert(x.computeNumberOfSpecies() == 0);
    ReactionSet y;
    assert(x == y);
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

    ReactionSet x(&reaction, &reaction + 1);
    assert(x.getSize() == 1);
    assert(x.getReaction(0) == reaction);
    assert(x.getBeginning() + 1 == x.getEnd());
    std::vector<double> populations;
    populations.push_back(3);
    assert(x.computePropensity(0, populations) ==
           RateConstant * populations[0]);
    /// CONTINUE REMOVE
    //assert(x.computeNumberOfSpecies() == 2);
  }

  return 0;
}

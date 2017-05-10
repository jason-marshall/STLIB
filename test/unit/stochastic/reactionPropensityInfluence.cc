// -*- C++ -*-

#include "stlib/stochastic/reactionPropensityInfluence.h"

using namespace stlib;

int
main()
{
  typedef stochastic::Reaction<true> Reaction;

  //-------------------------------------------------------------------------
  // Empty reaction.
  {
    std::cout << "Empty.\n";
    // Default constructor.
    Reaction x;
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence(0, &x, &x + 1, &influence,
        false);
    assert(influence.size() == 0);

    stochastic::computeReactionPropensityInfluence(0, &x, &x + 1, &influence,
        true);
    assert(influence.size() == 0);
  }
  //-------------------------------------------------------------------------
  // s0 -> s1
  {
    std::cout << "s0 -> s1.\n";
    std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
    reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
    products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
    std::vector<std::size_t> dependencies(1, 0);
    const double RateConstant = 2.0;
    Reaction x(reactants, products, dependencies, RateConstant);
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence(2, &x, &x + 1, &influence,
        false);
    assert(influence.size() == 0);

    stochastic::computeReactionPropensityInfluence(2, &x, &x + 1, &influence,
        true);
    assert(influence.getNumberOfArrays() == 1);
    assert(influence.size(0) == 1);
    assert(influence(0, 0) == 0);
  }
  //-------------------------------------------------------------------------
  // s0 -> s1, s2 -> s3
  {
    std::cout << "s0 -> s1, s2 -> s3.\n";
    std::vector<Reaction> reactions;
    // s0 -> s1
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    // s2 -> s3
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(2, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(3, 1));
      std::vector<std::size_t> dependencies(1, 2);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence
    (4, reactions.begin(), reactions.end(), &influence, false);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size() == 0);

    stochastic::computeReactionPropensityInfluence
    (4, reactions.begin(), reactions.end(), &influence, true);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 1);
    assert(influence(0, 0) == 0);
    assert(influence.size(1) == 1);
    assert(influence(1, 0) == 1);
  }
  //-------------------------------------------------------------------------
  // s0 -> s1, s1 -> s2
  {
    std::cout << "s0 -> s1, s1 -> s2.\n";
    std::vector<Reaction> reactions;
    // s0 -> s1
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    // s1 -> s2
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(2, 1));
      std::vector<std::size_t> dependencies(1, 1);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence
    (3, reactions.begin(), reactions.end(), &influence, false);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 1);
    assert(influence(0, 0) == 1);
    assert(influence.size(1) == 0);

    stochastic::computeReactionPropensityInfluence
    (3, reactions.begin(), reactions.end(), &influence, true);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 2);
    assert(influence(0, 0) == 0);
    assert(influence(0, 1) == 1);
    assert(influence.size(1) == 1);
    assert(influence(1, 0) == 1);
  }
  //-------------------------------------------------------------------------
  // s0 -> s1, s1 -> s0
  {
    std::cout << "s0 -> s1, s1 -> s0.\n";
    std::vector<Reaction> reactions;
    // s0 -> s1
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    // s1 -> s0
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      std::vector<std::size_t> dependencies(1, 1);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence
    (2, reactions.begin(), reactions.end(), &influence, false);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 1);
    assert(influence(0, 0) == 1);
    assert(influence.size(1) == 1);
    assert(influence(1, 0) == 0);

    stochastic::computeReactionPropensityInfluence
    (2, reactions.begin(), reactions.end(), &influence, true);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 2);
    assert(influence(0, 0) == 0);
    assert(influence(0, 1) == 1);
    assert(influence.size(1) == 2);
    assert(influence(1, 0) == 0);
    assert(influence(1, 1) == 1);
  }
  //-------------------------------------------------------------------------
  // s0 -> s1, s1 -> s1
  {
    std::cout << "s0 -> s1, s1 -> s1.\n";
    std::vector<Reaction> reactions;
    // s0 -> s1
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(0, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 0);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    // s1 -> s1
    {
      std::vector<std::pair<std::size_t, std::size_t> > reactants, products;
      reactants.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      products.push_back(std::make_pair<std::size_t, std::size_t>(1, 1));
      std::vector<std::size_t> dependencies(1, 1);
      const double RateConstant = 2.0;
      Reaction x(reactants, products, dependencies, RateConstant);
      reactions.push_back(x);
    }
    container::StaticArrayOfArrays<std::size_t> influence;

    stochastic::computeReactionPropensityInfluence
    (2, reactions.begin(), reactions.end(), &influence, false);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 1);
    assert(influence(0, 0) == 1);
    assert(influence.size(1) == 0);
    //std::cout << "s0 -> s1, s1 -> s1, false.\n" << influence;

    stochastic::computeReactionPropensityInfluence
    (2, reactions.begin(), reactions.end(), &influence, true);
    assert(influence.getNumberOfArrays() == 2);
    assert(influence.size(0) == 2);
    assert(influence(0, 0) == 0);
    assert(influence(0, 1) == 1);
    assert(influence.size(1) == 0);
    //std::cout << "s0 -> s1, s1 -> s1, true.\n" << influence;
  }

  return 0;
}

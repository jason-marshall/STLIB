// -*- C++ -*-

#include "stlib/stochastic/PropensitiesInhomogeneous.h"

#include <sstream>

using namespace stlib;

const std::size_t NumberOfReactions = 2;

// 0 -> x, x -> 0
void
computePropensities(std::vector<double>* propensities,
                    const std::vector<double>& populations, const double t)
{
  const double rate = 2. + std::sin(t);
  (*propensities)[0] = rate;
  (*propensities)[1] = rate * populations[0];
}

int
main()
{
  typedef stochastic::PropensitiesInhomogeneous<true> Propensities;
  typedef Propensities::ReactionSet ReactionSet;

  {
    // Default constructor.
    ReactionSet reactions;
    Propensities x(reactions);
    assert(x.getSize() == 0);
  }
  {
    // The reactions.
    ReactionSet reactions;
    {
      std::istringstream in("0 1 0 1 0  1 0 1 0 1 0");
      stochastic::readReactantsAndProductsAscii(in, NumberOfReactions,
          &reactions);
    }
    Propensities x(reactions);
    assert(x.getSize() == NumberOfReactions);
    std::vector<double> populations(1, 3.);
    x.set(populations, 0.);
    assert(x.propensities()[0] == 2.);
    assert(x.propensities()[1] == 6.);
    assert(x.sum() == 8.);
  }

  return 0;
}

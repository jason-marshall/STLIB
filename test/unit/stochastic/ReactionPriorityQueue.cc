// -*- C++ -*-

#include "stlib/stochastic/ReactionPriorityQueue.h"

using namespace stlib;

int
main()
{
  typedef stochastic::ReactionPriorityQueue<> ReactionPriorityQueue;
  typedef ReactionPriorityQueue::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  DiscreteUniformGenerator uniform;
  ReactionPriorityQueue x(10, &uniform);

  return 0;
}

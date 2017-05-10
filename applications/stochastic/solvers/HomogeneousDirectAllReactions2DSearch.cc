// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousDirectAllReactions_ipp__
#include "HomogeneousDirectAllReactions.ipp"
#undef __HomogeneousDirectAllReactions_ipp__

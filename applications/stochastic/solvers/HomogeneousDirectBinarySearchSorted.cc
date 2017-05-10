// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGeneratorBinarySearchSorted.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;

typedef numerical::DiscreteGeneratorBinarySearchSorted<> DiscreteGenerator;

#define STOCHASTIC_USE_INFLUENCE_IN_GENERATOR

#define __HomogeneousDirect_ipp__
#include "HomogeneousDirect.ipp"
#undef __HomogeneousDirect_ipp__

// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGeneratorLinearSearchDelayedUpdate.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;

typedef numerical::DiscreteGeneratorLinearSearchDelayedUpdate<> DiscreteGenerator;

#define __HomogeneousDirect_ipp__
#include "HomogeneousDirect.ipp"
#undef __HomogeneousDirect_ipp__

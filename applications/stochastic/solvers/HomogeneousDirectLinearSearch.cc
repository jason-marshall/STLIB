// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGeneratorLinearSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;

typedef numerical::DiscreteGeneratorLinearSearch<> DiscreteGenerator;

#define __HomogeneousDirect_ipp__
#include "HomogeneousDirect.ipp"
#undef __HomogeneousDirect_ipp__

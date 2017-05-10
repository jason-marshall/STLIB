// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearchSorted.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearchSorted<> DiscreteGenerator;

#define __HomogeneousDirect_ipp__
#include "HomogeneousDirect.ipp"
#undef __HomogeneousDirect_ipp__

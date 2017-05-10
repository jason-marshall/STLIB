// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousHistogramsMultiTimeDirect_ipp__
#include "HomogeneousHistogramsMultiTimeDirect.ipp"
#undef __HomogeneousHistogramsMultiTimeDirect_ipp__

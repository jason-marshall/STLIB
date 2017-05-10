// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousHistogramsAverageAps_ipp__
#include "HomogeneousHistogramsAverageAps.ipp"
#undef __HomogeneousHistogramsAverageAps_ipp__

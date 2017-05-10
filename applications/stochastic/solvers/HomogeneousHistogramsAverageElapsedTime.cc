// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousHistogramsAverageElapsedTime_ipp__
#include "HomogeneousHistogramsAverageElapsedTime.ipp"
#undef __HomogeneousHistogramsAverageElapsedTime_ipp__

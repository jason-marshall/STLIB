// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousHistogramsAverage_ipp__
#include "HomogeneousHistogramsAverage.ipp"
#undef __HomogeneousHistogramsAverage_ipp__

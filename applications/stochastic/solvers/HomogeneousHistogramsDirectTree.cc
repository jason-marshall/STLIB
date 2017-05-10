// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGenerator2DSearch.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
typedef numerical::DiscreteGenerator2DSearch<> DiscreteGenerator;

#define __HomogeneousHistogramsDirectTree_ipp__
#include "HomogeneousHistogramsDirectTree.ipp"
#undef __HomogeneousHistogramsDirectTree_ipp__

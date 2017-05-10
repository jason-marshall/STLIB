// -*- C++ -*-

#include "numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "numerical/random/discrete/DiscreteGeneratorBinarySearchRecursiveCdf.h"

typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;

typedef numerical::DiscreteGeneratorBinarySearchRecursiveCdf<>
DiscreteGenerator;

#define __HomogeneousDirect_ipp__
#include "HomogeneousDirect.ipp"
#undef __HomogeneousDirect_ipp__

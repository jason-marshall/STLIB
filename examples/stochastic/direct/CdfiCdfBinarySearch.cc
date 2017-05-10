// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorBinarySearch.h"

typedef numerical::DiscreteFiniteGeneratorBinarySearch<>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

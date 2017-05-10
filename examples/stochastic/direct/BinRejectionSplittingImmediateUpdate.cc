// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorBinsSplitting.h"

typedef numerical::DiscreteFiniteGeneratorBinsSplitting<true, true>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorRejectionBinsSplitting.h"

typedef numerical::DiscreteFiniteGeneratorRejectionBinsSplitting<true, false>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

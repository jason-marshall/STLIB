// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorCdfInversionUsingPartialRecursiveCdf.h"

typedef numerical::DiscreteFiniteGeneratorCdfInversionUsingPartialRecursiveCdf<true>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

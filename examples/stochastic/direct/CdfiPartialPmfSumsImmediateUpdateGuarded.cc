// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorCdfInversionUsingPartialPmfSums.h"

typedef numerical::DfgPmfWithGuard<> Pmf;
typedef numerical::DiscreteFiniteGeneratorCdfInversionUsingPartialPmfSums<0, true, Pmf>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

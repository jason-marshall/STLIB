// -*- C++ -*-

#include "common.h"
#include "numerical/random/discreteFinite/DiscreteFiniteGeneratorLinearSearch.h"

typedef numerical::DfgPmfWithGuard<> Pmf;
typedef numerical::TraitsForImmediateUpdate<true> Traits;
typedef numerical::DfgPmfAndSum<Pmf, Traits> PmfAndSum;
typedef numerical::DiscreteFiniteGeneratorLinearSearch<PmfAndSum>
DiscreteFiniteGenerator;

#define __stochastic_direct_main_ipp__
#include "../../../main.ipp"
#undef __stochastic_direct_main_ipp__

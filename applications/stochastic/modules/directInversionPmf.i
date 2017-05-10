// -*- C++ -*-

%module directInversionPmf

%{
#include "../../../src/stochastic/api.h"
#include "../../../src/numerical/random/discreteFinite/DiscreteFiniteGeneratorLinearSearch.h"
%}

%inline %{
typedef double Number;
typedef numerical::DfgPmfWithGuard<> Pmf;
typedef numerical::TraitsForImmediateUpdate<true> Traits;
typedef numerical::DfgPmfAndSum<Pmf, Traits> PmfAndSum;
typedef numerical::DiscreteFiniteGeneratorLinearSearch<PmfAndSum>
DiscreteFiniteGenerator;
typedef stochastic::Direct<DiscreteFiniteGenerator> Direct;
%}

%include "direct.ii"

// -*- C++ -*-

%module directInversionPartialCdf

%{
#include "../../../src/stochastic/api.h"
#include "../../../src/numerical/random/discreteFinite/DiscreteFiniteGeneratorCdfInversionUsingPartialPmfSums.h"
%}

%inline %{
typedef double Number;
typedef numerical::DfgPmfWithGuard<> Pmf;
typedef numerical::DiscreteFiniteGeneratorCdfInversionUsingPartialPmfSums<0, true, Pmf>
DiscreteFiniteGenerator;
typedef stochastic::Direct<DiscreteFiniteGenerator> Direct;
%}

%include "direct.ii"

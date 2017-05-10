// -*- C++ -*-

%module directRejectionBins

%{
#include "../../../src/stochastic/api.h"
#include "../../../src/numerical/random/discreteFinite/DiscreteFiniteGeneratorRejectionBinsSplitting.h"
%}

%inline %{
typedef double Number;
typedef numerical::DiscreteFiniteGeneratorRejectionBinsSplitting<true, true>
DiscreteFiniteGenerator;
typedef stochastic::Direct<DiscreteFiniteGenerator> Direct;
%}

%include "direct.ii"

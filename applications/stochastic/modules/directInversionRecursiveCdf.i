// -*- C++ -*-

%module directInversionRecursiveCdf

%{
#include "../../../src/stochastic/api.h"
#include "../../../src/numerical/random/discreteFinite/DiscreteFiniteGeneratorCdfInversionUsingPartialRecursiveCdf.h"
%}

%inline %{
typedef double Number;
typedef numerical::DiscreteFiniteGeneratorCdfInversionUsingPartialRecursiveCdf<true>
DiscreteFiniteGenerator;
typedef stochastic::Direct<DiscreteFiniteGenerator> Direct;
%}

%include "direct.ii"

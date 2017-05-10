// -*- C++ -*-

%module directInversionCdf

%{
#include "../../../src/stochastic/api.h"
#include "../../../src/numerical/random/discreteFinite/DiscreteFiniteGeneratorBinarySearch.h"
%}

%inline %{
typedef double Number;
typedef numerical::DiscreteFiniteGeneratorBinarySearch<>
DiscreteFiniteGenerator;
typedef stochastic::Direct<DiscreteFiniteGenerator> Direct;
%}

%include "direct.ii"

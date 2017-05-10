// -*- C++ -*-

const char* OutputName = "InversionRatioOfUniformsWinrand";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionRatioOfUniformsWinrand.h"

typedef numerical::PoissonGeneratorInversionRatioOfUniformsWinrand<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

// -*- C++ -*-

const char* OutputName = "InversionRejectionPatchworkWinrand";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionRejectionPatchworkWinrand.h"

typedef numerical::PoissonGeneratorInversionRejectionPatchworkWinrand<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

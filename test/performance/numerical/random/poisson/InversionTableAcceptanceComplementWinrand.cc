// -*- C++ -*-

const char* OutputName = "InversionTableAcceptanceComplementWinrand";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionTableAcceptanceComplementWinrand.h"

typedef numerical::PoissonGeneratorInversionTableAcceptanceComplementWinrand<>
Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

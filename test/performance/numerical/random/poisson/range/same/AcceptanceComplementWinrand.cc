// -*- C++ -*-

const char* OutputName = "AcceptanceComplementWinrand";

#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"

typedef stlib::numerical::PoissonGeneratorAcceptanceComplementWinrand<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

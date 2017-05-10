// -*- C++ -*-

const char* OutputName = "InversionBuildUpSimple";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionBuildUpSimple.h"

typedef stlib::numerical::PoissonGeneratorInversionBuildUpSimple<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

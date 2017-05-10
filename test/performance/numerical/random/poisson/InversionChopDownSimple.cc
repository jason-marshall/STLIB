// -*- C++ -*-

const char* OutputName = "InversionChopDownSimple";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDownSimple.h"

typedef numerical::PoissonGeneratorInversionChopDownSimple<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

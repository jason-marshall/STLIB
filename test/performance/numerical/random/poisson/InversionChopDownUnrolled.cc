// -*- C++ -*-

#define POISSON_MAX_MEAN_CONSTRUCTOR

const char* OutputName = "InversionChopDownUnrolled";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDownUnrolled.h"

typedef numerical::PoissonGeneratorInversionChopDownUnrolled<> Poisson;

#define CONSTRUCT(x, m)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform, m)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

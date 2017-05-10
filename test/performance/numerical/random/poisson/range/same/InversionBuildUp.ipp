// -*- C++ -*-

#ifndef __InversionBuildUp_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionBuildUp.h"

typedef stlib::numerical::PoissonGeneratorInversionBuildUp<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

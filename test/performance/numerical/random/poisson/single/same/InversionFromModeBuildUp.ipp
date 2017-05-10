// -*- C++ -*-

#ifndef __InversionFromModeBuildUp_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.h"

typedef stlib::numerical::PoissonGeneratorInversionFromModeBuildUp<> Poisson;

#define POISSON_SIZE_CONSTRUCTOR

#define CONSTRUCT(x, s)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform, s)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

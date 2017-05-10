// -*- C++ -*-

const char* OutputName = "DirectNr";

#include "stlib/numerical/random/poisson/PoissonGeneratorDirectNr.h"

typedef stlib::numerical::PoissonGeneratorDirectNr<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

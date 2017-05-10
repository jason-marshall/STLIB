// -*- C++ -*-

const char* OutputName = "StochKit";

#include "stlib/numerical/random/poisson/PoissonGeneratorStochKit.h"

typedef numerical::PoissonGeneratorStochKit<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

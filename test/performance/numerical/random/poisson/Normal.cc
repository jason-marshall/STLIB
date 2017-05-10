// -*- C++ -*-

const char* OutputName = "Normal";

#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

typedef numerical::PoissonGeneratorNormal<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

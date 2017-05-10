// -*- C++ -*-

const char* OutputName = "Ranlib";

#include "stlib/numerical/random/poisson/PoissonGeneratorRanlib.h"

typedef numerical::PoissonGeneratorRanlib<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

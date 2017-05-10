// -*- C++ -*-

const char* OutputName = "InversionTable";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionTable.h"

typedef stlib::numerical::PoissonGeneratorInversionTable<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

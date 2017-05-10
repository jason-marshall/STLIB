// -*- C++ -*-

const char* OutputName = "RejectionNr";

#include "stlib/numerical/random/poisson/PoissonGeneratorRejectionNr.h"

typedef stlib::numerical::PoissonGeneratorRejectionNr<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

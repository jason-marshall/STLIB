// -*- C++ -*-

const char* OutputName = "DirectRejectionNr";

#include "stlib/numerical/random/poisson/PoissonGeneratorDirectRejectionNr.h"

typedef stlib::numerical::PoissonGeneratorDirectRejectionNr<> Poisson;

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

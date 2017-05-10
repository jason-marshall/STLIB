// -*- C++ -*-

const char* OutputName = "ExponentialInterArrival";

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h"

typedef stlib::numerical::PoissonGeneratorExponentialInterArrival<> Poisson;

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson::ExponentialGenerator exponential(&uniform); \
Poisson x(&exponential)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

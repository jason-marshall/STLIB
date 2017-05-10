// -*- C++ -*-

const char* OutputName = "ExponentialInterArrivalUnrolled";

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrivalUnrolled.h"

typedef stlib::numerical::PoissonGeneratorExponentialInterArrivalUnrolled<> Poisson;

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson::ExponentialGenerator exponential(&uniform); \
Poisson x(&exponential)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

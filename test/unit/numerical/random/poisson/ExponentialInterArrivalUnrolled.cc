// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrivalUnrolled.h"

using namespace stlib;

typedef numerical::PoissonGeneratorExponentialInterArrivalUnrolled<>
PoissonGenerator;

const double Arguments[] = {0, 0.00001};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::ExponentialGenerator exponential(&uniform);
#define CONSTRUCT(x) PoissonGenerator x(&exponential)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

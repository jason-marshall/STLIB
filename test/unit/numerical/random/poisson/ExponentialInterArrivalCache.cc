// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h"

using namespace stlib;

typedef numerical::PoissonGeneratorExponentialInterArrival<> PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::ExponentialGenerator exponential(&uniform);
#define CONSTRUCT(x) PoissonGenerator x(&exponential)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

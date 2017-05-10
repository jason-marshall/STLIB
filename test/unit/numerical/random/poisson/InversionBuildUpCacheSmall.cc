// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN
#define NUMERICAL_POISSON_SMALL_MEAN

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionBuildUp.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInversionBuildUp<> PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100};

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x) PoissonGenerator x(&uniform)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

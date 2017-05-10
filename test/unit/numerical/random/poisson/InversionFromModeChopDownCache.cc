// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeChopDown.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInversionFromModeChopDown<> PoissonGenerator;

#define POISSON_SIZE_CONSTRUCTOR

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1000};

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x, s) PoissonGenerator x(&uniform, s)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

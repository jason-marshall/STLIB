// -*- C++ -*-

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInversionFromModeBuildUp<> PoissonGenerator;

#define POISSON_SIZE_CONSTRUCTOR

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100};

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x, s) PoissonGenerator x(&uniform, s)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

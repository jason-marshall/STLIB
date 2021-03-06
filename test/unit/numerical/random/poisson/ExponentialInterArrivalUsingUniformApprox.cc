// -*- C++ -*-

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrivalUsingUniform.h"

using namespace stlib;

typedef numerical::PoissonGeneratorExponentialInterArrivalUsingUniform<>
PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 20};

#define POISSON_MAX_MEAN_CONSTRUCTOR

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x, m) PoissonGenerator x(&uniform, m)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

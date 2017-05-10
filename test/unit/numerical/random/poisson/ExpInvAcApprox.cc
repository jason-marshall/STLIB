// -*- C++ -*-

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION

#include "stlib/numerical/random/poisson/PoissonGeneratorExpInvAc.h"

using namespace stlib;

typedef numerical::PoissonGeneratorExpInvAc<> PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::ExponentialGenerator exponential(&uniform);
static PoissonGenerator::NormalGenerator normal(&uniform);
#define CONSTRUCT(x) PoissonGenerator x(&exponential, &normal)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

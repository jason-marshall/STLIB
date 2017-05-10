// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorExpAcNorm.h"

using namespace stlib;

typedef numerical::PoissonGeneratorExpAcNorm<> PoissonGenerator;

#define POISSON_THRESHHOLD_CONSTRUCTOR

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::ExponentialGenerator exponential(&uniform);
static PoissonGenerator::NormalGenerator normal(&uniform);
#define CONSTRUCT(x, t) PoissonGenerator x(&exponential, &normal, t)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInvAcNormSure<> PoissonGenerator;

#define POISSON_DOUBLE_THRESHHOLD_CONSTRUCTOR

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::NormalGenerator normal(&uniform);
#define CONSTRUCT(x, t1, t2) PoissonGenerator x(&normal, t1, t2)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

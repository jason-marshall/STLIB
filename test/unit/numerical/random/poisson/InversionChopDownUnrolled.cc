// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDownUnrolled.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInversionChopDownUnrolled<> PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100};

#define POISSON_MAX_MEAN_CONSTRUCTOR

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x, m) PoissonGenerator x(&uniform, m)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

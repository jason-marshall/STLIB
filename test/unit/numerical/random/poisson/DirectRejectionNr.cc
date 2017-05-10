// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorDirectRejectionNr.h"

using namespace stlib;

typedef numerical::PoissonGeneratorDirectRejectionNr<>
PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x) PoissonGenerator x(&uniform)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

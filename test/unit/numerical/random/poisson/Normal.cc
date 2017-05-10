// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

using namespace stlib;

typedef numerical::PoissonGeneratorNormal<> PoissonGenerator;

const double Arguments[] = {100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
static PoissonGenerator::NormalGenerator normal(&uniform);
#define CONSTRUCT(x) PoissonGenerator x(&normal)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

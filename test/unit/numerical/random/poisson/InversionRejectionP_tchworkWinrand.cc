// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionRejectionPatchworkWinrand.h"

using namespace stlib;

typedef numerical::PoissonGeneratorInversionRejectionPatchworkWinrand<>
PoissonGenerator;

const double Arguments[] = {0, 0.01, 0.1, 1, 10, 100, 1e4, 1e8};

static PoissonGenerator::DiscreteUniformGenerator uniform;
#define CONSTRUCT(x) PoissonGenerator x(&uniform)

#define __test_numerical_random_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_main_ipp__

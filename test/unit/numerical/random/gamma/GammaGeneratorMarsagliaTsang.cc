// -*- C++ -*-

#include "stlib/numerical/random/gamma/GammaGeneratorMarsagliaTsang.h"

using namespace stlib;

typedef numerical::GammaGeneratorMarsagliaTsang<> GammaGenerator;

const double Arguments[] = {1, 2, 4, 8};

#define __test_numerical_random_gamma_main_ipp__
#include "main.ipp"
#undef __test_numerical_random_gamma_main_ipp__

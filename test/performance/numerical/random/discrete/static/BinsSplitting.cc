// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorBinsSplitting.h"

typedef stlib::numerical::DiscreteGeneratorBinsSplitting<false> Generator;

#define NUMERICAL_SET_INDEX_BITS

#define __performance_numerical_random_discrete_main_ipp__
#include "main.ipp"
#undef __performance_numerical_random_discrete_main_ipp__

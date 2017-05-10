// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorBinarySearchSorted.h"

typedef stlib::numerical::DiscreteGeneratorBinarySearchSorted<> Generator;

#define NUMERICAL_USE_INFLUENCE
#define NUMERICAL_REBUILD

#define __performance_numerical_random_discrete_main_ipp__
#include "main.ipp"
#undef __performance_numerical_random_discrete_main_ipp__


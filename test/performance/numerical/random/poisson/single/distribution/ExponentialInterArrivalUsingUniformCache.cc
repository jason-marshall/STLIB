// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN

const char* OutputName = "ExponentialInterArrivalUsingUniformCache";

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson x(&uniform)

#define __ExponentialInterArrivalUsingUniform_ipp__
#include "ExponentialInterArrivalUsingUniform.ipp"
#undef __ExponentialInterArrivalUsingUniform_ipp__

// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN
#define NUMERICAL_POISSON_SMALL_MEAN

const char* OutputName = "ExponentialInterArrivalUsingUniformCacheSmall";

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson x(&uniform)

#define __ExponentialInterArrivalUsingUniform_ipp__
#include "ExponentialInterArrivalUsingUniform.ipp"
#undef __ExponentialInterArrivalUsingUniform_ipp__

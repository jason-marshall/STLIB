// -*- C++ -*-

const char* OutputName = "ExponentialInterArrivalUsingUniform";

#define CONSTRUCT(x) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson x(&uniform)

#define __ExponentialInterArrivalUsingUniform_ipp__
#include "ExponentialInterArrivalUsingUniform.ipp"
#undef __ExponentialInterArrivalUsingUniform_ipp__

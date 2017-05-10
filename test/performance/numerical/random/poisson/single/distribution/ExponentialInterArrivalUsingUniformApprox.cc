// -*- C++ -*-

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
#define POISSON_MAX_MEAN_CONSTRUCTOR

const char* OutputName = "ExponentialInterArrivalUsingUniformApprox";

#define CONSTRUCT(x, m)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform, m)

#define __ExponentialInterArrivalUsingUniform_ipp__
#include "ExponentialInterArrivalUsingUniform.ipp"
#undef __ExponentialInterArrivalUsingUniform_ipp__

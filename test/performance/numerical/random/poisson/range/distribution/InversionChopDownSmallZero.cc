// -*- C++ -*-

#define NUMERICAL_POISSON_SMALL_MEAN
#define NUMERICAL_POISSON_ZERO_MEAN

const char* OutputName = "InversionChopDownSmallZero";

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __InversionChopDown_ipp__
#include "InversionChopDown.ipp"
#undef __InversionChopDown_ipp__

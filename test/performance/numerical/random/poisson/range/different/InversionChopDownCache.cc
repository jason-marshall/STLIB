// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN

const char* OutputName = "InversionChopDownCache";

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __InversionChopDown_ipp__
#include "InversionChopDown.ipp"
#undef __InversionChopDown_ipp__

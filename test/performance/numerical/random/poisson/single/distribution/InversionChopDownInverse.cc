// -*- C++ -*-

#define NUMERICAL_POISSON_STORE_INVERSE

const char* OutputName = "InversionChopDownInverse";

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __InversionChopDown_ipp__
#include "InversionChopDown.ipp"
#undef __InversionChopDown_ipp__

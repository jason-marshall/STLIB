// -*- C++ -*-

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
#define POISSON_MAX_MEAN_CONSTRUCTOR

const char* OutputName = "InversionChopDownApprox";

#define CONSTRUCT(x, m)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform, m)

#define __InversionChopDown_ipp__
#include "InversionChopDown.ipp"
#undef __InversionChopDown_ipp__

// -*- C++ -*-

const char* OutputName = "ExpInvAc";

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
#include "stlib/numerical/random/poisson/PoissonGeneratorExpInvAc.h"

typedef stlib::numerical::PoissonGeneratorExpInvAc<> Poisson;

#define CONSTRUCT(x)					\
  Poisson::DiscreteUniformGenerator uniform;		\
  Poisson::ExponentialGenerator exponential(&uniform);	\
  Poisson::NormalGenerator normal(&uniform);		\
  Poisson x(&exponential, &normal)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

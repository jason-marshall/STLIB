// -*- C++ -*-

const char* OutputName = "InvAcNorm";

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNorm.h"

typedef stlib::numerical::PoissonGeneratorInvAcNorm<> Poisson;

#define POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR

#define CONSTRUCT(x, t)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson::NormalGenerator normal(&uniform);	\
  Poisson x(&normal, t)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

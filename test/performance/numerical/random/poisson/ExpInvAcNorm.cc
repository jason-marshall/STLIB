// -*- C++ -*-

const char* OutputName = "ExpInvAcNorm";

#define NUMERICAL_POISSON_HERMITE_APPROXIMATION
#include "stlib/numerical/random/poisson/PoissonGeneratorExpInvAcNorm.h"

typedef numerical::PoissonGeneratorExpInvAcNorm<> Poisson;

#define POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR

#define CONSTRUCT(x, t) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson::ExponentialGenerator exponential(&uniform); \
Poisson::NormalGenerator normal(&uniform); \
Poisson x(&exponential, &normal, t)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

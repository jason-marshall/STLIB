// -*- C++ -*-

const char* OutputName = "ExpAcNorm";

#include "stlib/numerical/random/poisson/PoissonGeneratorExpAcNorm.h"

typedef stlib::numerical::PoissonGeneratorExpAcNorm<> Poisson;

#define POISSON_NORMAL_THRESHHOLD_CONSTRUCTOR

#define CONSTRUCT(x, t) \
Poisson::DiscreteUniformGenerator uniform; \
Poisson::ExponentialGenerator exponential(&uniform); \
Poisson::NormalGenerator normal(&uniform); \
Poisson x(&exponential, &normal, t)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

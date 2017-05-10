// -*- C++ -*-

const char* OutputName = "InversionCheckPdf";

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionCheckPdf.h"

typedef numerical::PoissonGeneratorInversionCheckPdf<> Poisson;

#define CONSTRUCT(x)				\
  Poisson::DiscreteUniformGenerator uniform;	\
  Poisson x(&uniform)

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

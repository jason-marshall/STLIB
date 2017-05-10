// -*- C++ -*-

const char* OutputName = "MarsagliaTsang";

#include "stlib/numerical/random/gamma/GammaGeneratorMarsagliaTsang.h"

typedef stlib::numerical::GammaGeneratorMarsagliaTsang<> Gamma;

#define __main_ipp__
#include "main.ipp"
#undef __main_ipp__

// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN
#define NUMERICAL_POISSON_SMALL_MEAN

const char* OutputName = "InversionFromModeChopDownCacheSmall";

#define __InversionFromModeChopDown_ipp__
#include "InversionFromModeChopDown.ipp"
#undef __InversionFromModeChopDown_ipp__

// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN
#define NUMERICAL_POISSON_SMALL_MEAN

const char* OutputName = "InversionBuildUpCacheSmall";

#define __InversionBuildUp_ipp__
#include "InversionBuildUp.ipp"
#undef __InversionBuildUp_ipp__

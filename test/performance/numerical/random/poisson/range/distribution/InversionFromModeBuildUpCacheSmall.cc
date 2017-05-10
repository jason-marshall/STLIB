// -*- C++ -*-

#define NUMERICAL_POISSON_CACHE_OLD_MEAN
#define NUMERICAL_POISSON_SMALL_MEAN

const char* OutputName = "InversionFromModeBuildUpCacheSmall";

#define __InversionFromModeBuildUp_ipp__
#include "InversionFromModeBuildUp.ipp"
#undef __InversionFromModeBuildUp_ipp__

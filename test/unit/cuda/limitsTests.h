// -*- C++ -*-

#include "stlib/cuda/limits.h"

#ifdef __CUDA_ARCH__
__device__
#endif
bool
limitsFloat()
{
  if (!(std::numeric_limits<float>::min() > 0)) {
    return false;
  }
  if (!(std::numeric_limits<float>::epsilon() > 0)) {
    return false;
  }
  if (!(std::numeric_limits<float>::max() <
        std::numeric_limits<float>::infinity())) {
    return false;
  }
  if (std::numeric_limits<float>::quiet_NaN() ==
      std::numeric_limits<float>::quiet_NaN()) {
    return false;
  }
  return true;
}

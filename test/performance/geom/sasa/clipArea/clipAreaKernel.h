/* -*- C -*- */

#include "Ball.h"

const size_t ThreadsPerBlock = 64;

extern "C"
void
clipKernel(const float* referenceMesh, const unsigned meshSize,
           const float3* centers, unsigned centersSize,
           const unsigned* delimiters, const float3* clippingCenters,
           unsigned* activeCounts);

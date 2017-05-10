/* -*- C -*- */

#ifndef __test_performance_geom_sasa_clipArea_clipAreaGpu_h__
#define __test_performance_geom_sasa_clipArea_clipAreaGpu_h__

#include "Ball.h"

#include <vector>

std::size_t
calculateAreaGpu(const std::vector<float3>& referenceMesh,
                 const std::vector<float3>& centers,
                 const std::vector<std::size_t>& clippingSizes,
                 const std::vector<std::size_t>& clippingIndices);

#endif

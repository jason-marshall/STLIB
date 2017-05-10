/* -*- C -*- */

#include "clipAreaCpu.h"

#include <numeric>

namespace
{

#if 0
bool
isInside(const Ball& ball, const float3& p)
{
  return (ball.center.x - p.x) * (ball.center.x - p.x) +
         (ball.center.y - p.y) * (ball.center.y - p.y) +
         (ball.center.z - p.z) * (ball.center.z - p.z) < ball.squaredRadius;
}
#endif

bool
isInside(const float3& center, const float3& p)
{
  return (center.x - p.x) * (center.x - p.x) +
         (center.y - p.y) * (center.y - p.y) +
         (center.z - p.z) * (center.z - p.z) < 1;
}

}

std::size_t
calculateAreaCpu(const std::vector<float3>& referenceMesh,
                 const std::vector<float3>& centers,
                 const std::vector<std::size_t>& clippingSizes,
                 const std::vector<std::size_t>& clippingIndices)
{
  // Calculate the clipped area for each ball.
  std::vector<std::size_t> activeCounts(centers.size());
  float3 p;
  std::size_t start = 0;
  // For each ball.
  for (std::size_t i = 0; i != centers.size(); ++i) {
    const float3& center = centers[i];
    // Count the clipped points.
    std::size_t clippedCount = 0;
    // Loop over the mesh points.
    for (std::size_t j = 0; j != referenceMesh.size(); ++j) {
      // Translate the reference mesh point to lie on the ball.
      p = referenceMesh[j];
      p.x += center.x;
      p.y += center.y;
      p.z += center.z;
      // Loop over the clipping balls to determine if the point is clipped.
      for (std::size_t k = start; k != start + clippingSizes[i]; ++k) {
        if (isInside(centers[clippingIndices[k]], p)) {
          ++clippedCount;
          break;
        }
      }
    }
    activeCounts[i] = referenceMesh.size() - clippedCount;
    start += clippingSizes[i];
  }
  return std::accumulate(activeCounts.begin(), activeCounts.end(),
                         std::size_t(0));
}

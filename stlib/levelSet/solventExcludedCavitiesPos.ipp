/* -*- C++ -*- */

#if !defined(__levelSet_solventExcludedCavitiesPos_ipp__)
#error This file is an implementation detail of solventExcludedCavitiesPos.
#endif

namespace stlib
{
namespace levelSet
{


//! Distribute points on a sphere with the golden section spiral algorithm.
/*!
  http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere
*/
inline
void
distributePointsOnSphereWithGoldenSectionSpiral
#ifdef STLIB_NO_SIMD_INTRINSICS
(std::vector<std::array<float, 3> >* points)
#else
(container::vector<__m128>* points)
#endif
{
  // Check the singular case.
  if (points->size() == 0) {
    return;
  }
  // Use double precision for internal calculations.
  const double Delta = numerical::Constants<double>::Pi() *
                       (3. - std::sqrt(5.));
  double longitude = 0;
  double dz = 2.0 / points->size();
  double z = 1. - dz / 2;
  double r;
  for (std::size_t i = 0; i != points->size(); ++i) {
    r = std::sqrt(1. - z * z);
#ifdef STLIB_NO_SIMD_INTRINSICS
    (*points)[i][0] = r * std::cos(longitude);
    (*points)[i][1] = r * std::sin(longitude);
    (*points)[i][2] = z;
#else
    (*points)[i] = _mm_set_ps(0, z, r * std::sin(longitude),
                              r * std::cos(longitude));
#endif
    z -= dz;
    longitude += Delta;
  }
}


// Return true if the probe intersects any of the balls that are close to it.
// Note the sequence of close balls must not include the current ball.
inline
bool
doesIntersect(geom::Ball<float, 3> probe,
              const std::vector<geom::Ball<float, 3> >& close)
{
  for (std::size_t i = 0; i != close.size(); ++i) {
    if (doIntersect(probe, close[i])) {
      return true;
    }
  }
  return false;
}


// Return true if the probe intersects any of the balls that are close to it.
// Note the sequence of close balls must not include the current ball.
// Each element of the close vector holds the packed coordinates for four
// close balls.
inline
bool
doesIntersect(const __m128 center, const __m128 radius,
              const std::vector<std::array<__m128, 4> >& close)
{
  const float* p = reinterpret_cast<const float*>(&center);
  __m128 x = _mm_set1_ps(p[0]);
  __m128 y = _mm_set1_ps(p[1]);
  __m128 z = _mm_set1_ps(p[2]);

  for (std::size_t i = 0; i != close.size(); ++i) {
    // Test four at a time.
    if (_mm_movemask_ps(_mm_cmple_ps((x - close[i][0]) * (x - close[i][0]) +
                                     (y - close[i][1]) * (y - close[i][1]) +
                                     (z - close[i][2]) * (z - close[i][2]),
                                     (radius + close[i][3]) *
                                     (radius + close[i][3])))) {
      return true;
    }
  }
  return false;
}


class ClipByProbesBase
{
protected:
  typedef geom::Ball<float, 3> Ball;

  // CONTINUE: Choose a suitable spacing for each radius.
  BOOST_STATIC_CONSTEXPR std::size_t NumPointsOnSphere = 1024;

  const float _probeRadius;
  std::vector<Ball> _closeToBall;

  ClipByProbesBase(const float probeRadius) :
    _probeRadius(probeRadius),
    _closeToBall()
  {
  }

  // Make a list of the balls that are close enough to the specified
  // ball to intersect one of the probes placed on its surface.
  // Note that we do not include the current ball in the sequence
  // of close balls.
  void
  determineCloseToBall(const Ball& ball,
                       const std::vector<Ball>& influencing)
  {
    _closeToBall.clear();
    const float t = ball.radius + 2 * _probeRadius;
    for (std::size_t k = 0; k != influencing.size(); ++k) {
      const float d2 = ext::squaredDistance(ball.center,
                                            influencing[k].center);
      if (d2 != 0 && d2 <
          (t + influencing[k].radius) * (t + influencing[k].radius)) {
        _closeToBall.push_back(influencing[k]);
      }
    }
  }

};

#ifdef STLIB_NO_SIMD_INTRINSICS

class ClipByProbes :
  public ClipByProbesBase
{

  typedef std::array<float, 3> Point;

  std::vector<Point> _pointsOnSphere;

public:

  ClipByProbes(const float probeRadius) :
    ClipByProbesBase(probeRadius),
    _pointsOnSphere(NumPointsOnSphere)
  {
    distributePointsOnSphereWithGoldenSectionSpiral(&_pointsOnSphere);
  }

  void
  operator()(PatchActive* patch, const Ball& patchBall,
             const std::vector<Ball>& closeToPatch,
             const std::vector<Ball>& influencing)
  {
    // Loop over the balls that are close to the patch and clip by the
    // probes that are positioned on each ball.
    for (std::size_t j = 0; j != closeToPatch.size(); ++j) {
      // If there are no active grid points, we may skip further
      // calculations.
      if (! patch->hasActive()) {
        break;
      }
      _clip(patch, patchBall, closeToPatch[j], influencing);
    }
  }

private:

  void
  _clip(PatchActive* patch, const Ball& patchBall,
        const Ball& closeToPatch,
        const std::vector<Ball>& influencing)
  {
    const float threshold = patchBall.radius + _probeRadius;
    const float threshSquared = threshold * threshold;

    determineCloseToBall(closeToPatch, influencing);

    // Set the probe radius. (Later we'll set the center for each probe.)
    Ball probe = {{{0, 0, 0}}, _probeRadius};
    // Loop over the probes that are placed on the surface of the current ball.
    for (std::size_t k = 0; k != _pointsOnSphere.size(); ++k) {
      // Position the probe.
      probe.center = closeToPatch.center +
                     (closeToPatch.radius + _probeRadius) * _pointsOnSphere[k];
      // First reject the probe positions that are too far from the
      // patch to affect it.
      if (ext::squaredDistance(patchBall.center, probe.center) >
          threshSquared) {
        continue;
      }
      // Next reject the probes that are not valid, i.e. ones that
      // intersect an influencing ball.
      if (doesIntersect(probe, _closeToBall)) {
        continue;
      }
      // Clip by the probe.
      patch->clip(probe);
    }
  }
};

#else

class ClipByProbes :
  public ClipByProbesBase
{

  container::vector<__m128> _pointsOnSphere;
  // While _closeToBall stores ball in the format
  // xyzr xyzr xyzr xyzr ...,
  // _closeToBallPacked stores the information in the order
  // xxxx yyyy zzzz rrrr ....
  // This enables more efficient SIMD calculations.
  std::vector<std::array<__m128, 4> > _closeToBallPacked;

public:

  ClipByProbes(const float probeRadius) :
    ClipByProbesBase(probeRadius),
    _pointsOnSphere(NumPointsOnSphere),
    _closeToBallPacked()
  {
    distributePointsOnSphereWithGoldenSectionSpiral(&_pointsOnSphere);
  }

  void
  operator()(PatchActive* patch, const Ball& patchBall,
             const std::vector<Ball>& closeToPatch,
             const std::vector<Ball>& influencing)
  {
    // Loop over the balls that are close to the patch and clip by the
    // probes that are positioned on each ball.
    for (std::size_t j = 0; j != closeToPatch.size(); ++j) {
      // If there are no active grid points, we may skip further
      // calculations.
      if (! patch->hasActive()) {
        break;
      }
      _clip(patch, patchBall, closeToPatch[j], influencing);
    }
  }

private:

  void
  _clip(PatchActive* patch, const Ball& patchBall,
        const Ball& closeToPatch,
        const std::vector<Ball>& influencing)
  {
    const float threshold = patchBall.radius + _probeRadius;

    determineCloseToBall(closeToPatch, influencing);

    // Set the probe radius. (Later we'll set the center for each probe.)
    Ball probe = {{{0, 0, 0}}, _probeRadius};
    // Pad the set of balls that are close to the current so that the
    // size is a multiple of four.
    const float Inf = std::numeric_limits<float>::infinity();
    const Ball Dummy = {{{Inf, Inf, Inf}}, 0};
    _closeToBall.insert(_closeToBall.end(), (4 - _closeToBall.size() % 4) % 4,
                        Dummy);
    // Pack the coordinates and radii.
    _closeToBallPacked.resize(_closeToBall.size() / 4);
    for (std::size_t k = 0; k != _closeToBall.size(); k += 4) {
      for (std::size_t n = 0; n != 3; ++n) {
        _closeToBallPacked[k / 4][n] =
          _mm_set_ps(_closeToBall[k + 3].center[n],
                     _closeToBall[k + 2].center[n],
                     _closeToBall[k + 1].center[n],
                     _closeToBall[k].center[n]);
      }
      _closeToBallPacked[k / 4][3] =
        _mm_set_ps(_closeToBall[k + 3].radius,
                   _closeToBall[k + 2].radius,
                   _closeToBall[k + 1].radius,
                   _closeToBall[k].radius);
    }

    const __m128 patchBallCenter = _mm_set_ps(0, patchBall.center[2],
                                   patchBall.center[1],
                                   patchBall.center[0]);
    const __m128 closeToPatchCenter =
      _mm_set_ps(0, closeToPatch.center[2],
                 closeToPatch.center[1],
                 closeToPatch.center[0]);
    const __m128 offsetRadius =
      _mm_set1_ps(closeToPatch.radius + _probeRadius);
    // Note that it is better for the performance to define these
    // constants here instead of in an outer loop.
    const __m128 probeRadius128 = _mm_set1_ps(_probeRadius);
    const __m128 threshSquaredScalar = _mm_set_ps(0, 0, 0, threshold *
                                       threshold);
    __m128 d;
    __m128 probeCenter;
    // Loop over the probes that are placed on surface of the current ball.
    for (std::size_t k = 0; k != _pointsOnSphere.size(); ++k) {
      // Position the probe.
      probeCenter = closeToPatchCenter + offsetRadius * _pointsOnSphere[k];
      // First reject the probe positions that are too far from the
      // patch to affect it.
      d = patchBallCenter - probeCenter;
      if (_mm_comigt_ss(simd::dot(d, d), threshSquaredScalar)) {
        continue;
      }
      // Next reject the probes that are not valid, i.e. ones that
      // intersect an influencing ball.
      if (doesIntersect(probeCenter, probeRadius128, _closeToBallPacked)) {
        continue;
      }
      // Clip by the probe.
      probe.center[0] = reinterpret_cast<const float*>(&probeCenter)[0];
      probe.center[1] = reinterpret_cast<const float*>(&probeCenter)[1];
      probe.center[2] = reinterpret_cast<const float*>(&probeCenter)[2];
      patch->clip(probe);
    }
  }
};

#endif

inline
float
solventExcludedCavitiesPos
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies)
{
  typedef container::SimpleMultiIndexExtentsIterator<3> Iterator;
  typedef std::array<float, 3> Point;
  typedef geom::Ball<float, 3> Ball;

  assert(dependencies.getNumberOfArrays() == ext::product(grid.gridExtents));

  // Set the radius of the ball that covers the patch. (Later we'll set
  // the center for each patch.)
  Ball patchBall = {{{0, 0, 0}}, float((PatchExtent - 1) * 0.5f *
    grid.spacing * sqrt(3.f))
  };
  // The threshold for computing distance for a probe is the radius
  // of the patch plus the probe radius. If the center of the probe
  // is farther than this from the center of the patch, the probe
  // does not touch the patch and there is no need to compute
  // distances. We also use this in determining for which balls to
  // compute the distance.
  const float threshold = patchBall.radius + probeRadius;

  ClipByProbes clipByProbes(probeRadius);

  PatchDistance distance(grid.spacing);
  PatchActive patch(grid.spacing);
  std::size_t count = 0;
  std::vector<Ball> influencing, closeToPatch;
  std::vector<float> squaredDistances;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.gridExtents);
  for (Iterator i = Iterator::begin(grid.gridExtents); i != end; ++i) {
    // The lower corner of the patch.
    const Point lowerCorner = grid.getPatchLowerCorner(*i);
    // The center of the patch.
    patchBall.center =
      lowerCorner + (PatchExtent - 1) * 0.5f * grid.spacing;

    // Make a copy of the influencing balls, which are within a distance
    // of twice the probe radius from the patch.
    {
      const std::size_t index = grid.arrayIndex(*i);
      influencing.clear();
      for (std::size_t j = 0; j != dependencies.size(index); ++j) {
        influencing.push_back(balls[dependencies(index, j)]);
      }
    }
    // From the influencing balls, select the ones that are within a
    // distance of the probe radius to the patch.  If the ball is
    // not close to the patch, we don't need to compute the
    // Euclidean distances. (We only need to compute the distance up
    // to the probe radius.)  If the ball is further than
    // probeRadius from the patch, we do not need to clip by any of
    // the probes on its surface. This clipping will be done when we
    // clip by unobstructed probes.
    closeToPatch.clear();
    for (std::size_t j = 0; j != influencing.size(); ++j) {
      if (ext::squaredDistance(patchBall.center, influencing[j].center) <
          (threshold + influencing[j].radius) *
          (threshold + influencing[j].radius)) {
        closeToPatch.push_back(influencing[j]);
      }
    }
    // If there are no balls that are close to the patch, there is no
    // contribution to the SEC.
    if (closeToPatch.empty()) {
      continue;
    }

    // Sort the balls by the squared distance from the center of the patch.
    // This improves the performance of determining which placed probes
    // are valid.
    squaredDistances.resize(influencing.size());
    for (std::size_t k = 0; k != squaredDistances.size(); ++k) {
      squaredDistances[k] = ext::squaredDistance(patchBall.center,
                                                 influencing[k].center);
    }
    ads::sortTogether(squaredDistances.begin(), squaredDistances.end(),
                      influencing.begin(), influencing.end());

    // Initialize the distance patch.
    distance.initialize(lowerCorner);
    // Compute the Euclidean distance to the influencing balls.
    for (std::size_t j = 0; j != closeToPatch.size(); ++j) {
      distance.unionEuclidean(closeToPatch[j]);
    }

    // If a distance is greater than the probe radius, then the point
    // is covered by an unobstructed probe. Thus, the distance is negative.
    // For these cases, set the distance to a negative value.
    distance.conditionalSetValueGe(probeRadius, -1.f);

    // Initialize the patch.
    patch.initializePositive(grid.getPatchLowerCorner(*i), distance.grid);
    // Loop over the balls that are close to the patch and clip by the
    // probes that are positioned on each ball.
    clipByProbes(&patch, patchBall, closeToPatch, influencing);
    // Add the number of positive grid points in this patch.
    count += patch.numActivePoints();
  }

  // Return the total volume.
  return count * grid.spacing * grid.spacing * grid.spacing;
}


inline
void
solventExcludedCavitiesPos
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies,
 std::vector<float>* volumes)
{
  typedef container::SimpleMultiIndexExtentsIterator<3> Iterator;
  typedef std::array<float, 3> Point;
  typedef geom::Ball<float, 3> Ball;

#ifdef STLIB_NO_SIMD_INTRINSICS
  const std::size_t VectorSize = 1;
#elif defined(__AVX2__)
  const std::size_t VectorSize = 8;
#else
  const std::size_t VectorSize = 4;
#endif

  assert(dependencies.getNumberOfArrays() == ext::product(grid.gridExtents));

  // Initialize the volumes for each ball.
  volumes->resize(balls.size());
  std::fill(volumes->begin(), volumes->end(), float(0));

  // Set the radius of the ball that covers the patch. (Later we'll set
  // the center for each patch.)
  Ball patchBall = {{{0, 0, 0}}, float((PatchExtent - 1) * 0.5f *
    grid.spacing * sqrt(3.f))
  };
  // The threshold for computing distance for a probe is the radius
  // of the patch plus the probe radius. If the center of the probe
  // is farther than this from the center of the patch, the probe
  // does not touch the patch and there is no need to compute
  // distances. We also use this in determining for which balls to
  // compute the distance.
  const float threshold = patchBall.radius + probeRadius;
  const float voxelVolume = grid.spacing * grid.spacing * grid.spacing;

  ClipByProbes clipByProbes(probeRadius);

  PatchDistanceIdentifier distance(grid.spacing);
  PatchActive patch(grid.spacing);
  std::vector<Ball> influencing, closeToPatch;
  std::vector<unsigned> influencingIds, closeToPatchIds;
  std::vector<float> squaredDistances;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.gridExtents);
  for (Iterator i = Iterator::begin(grid.gridExtents); i != end; ++i) {
    // The lower corner of the patch.
    const Point lowerCorner = grid.getPatchLowerCorner(*i);
    // The center of the patch.
    patchBall.center =
      lowerCorner + (PatchExtent - 1) * 0.5f * grid.spacing;

    // Make a copy of the influencing balls, which are within a distance
    // of twice the probe radius from the patch.
    {
      const std::size_t index = grid.arrayIndex(*i);
      influencing.clear();
      influencingIds.clear();
      for (std::size_t j = 0; j != dependencies.size(index); ++j) {
        const unsigned id = dependencies(index, j);
        influencing.push_back(balls[id]);
        influencingIds.push_back(id);
      }
    }
    // From the influencing balls, select the ones that are within a
    // distance of the probe radius to the patch.  If the ball is
    // not close to the patch, we don't need to compute the
    // Euclidean distances. (We only need to compute the distance up
    // to the probe radius.)  If the ball is further than
    // probeRadius from the patch, we do not need to clip by any of
    // the probes on its surface. This clipping will be done when we
    // clip by unobstructed probes.
    closeToPatch.clear();
    closeToPatchIds.clear();
    for (std::size_t j = 0; j != influencing.size(); ++j) {
      if (ext::squaredDistance(patchBall.center, influencing[j].center) <
          (threshold + influencing[j].radius) *
          (threshold + influencing[j].radius)) {
        closeToPatch.push_back(influencing[j]);
        closeToPatchIds.push_back(influencingIds[j]);
      }
    }
    // If there are no balls that are close to the patch, there is no
    // contribution to the SEC.
    if (closeToPatch.empty()) {
      continue;
    }

    // Sort the balls by the squared distance from the center of the patch.
    // This improves the performance of determining which placed probes
    // are valid. Note that we no longer use influencingIds, so we don't
    // need to sort that sequence.
    squaredDistances.resize(influencing.size());
    for (std::size_t k = 0; k != squaredDistances.size(); ++k) {
      squaredDistances[k] = ext::squaredDistance(patchBall.center,
                                                 influencing[k].center);
    }
    ads::sortTogether(squaredDistances.begin(), squaredDistances.end(),
                      influencing.begin(), influencing.end());

    // Initialize the distance patch.
    distance.initialize(lowerCorner);
    // Compute the Euclidean distance to the influencing balls.
    for (std::size_t j = 0; j != closeToPatch.size(); ++j) {
      distance.unionEuclidean(closeToPatch[j], closeToPatchIds[j]);
    }

    // If a distance is greater than the probe radius, then the point
    // is covered by an unobstructed probe. Thus, the distance is negative.
    // For these cases, set the distance to a negative value.
    distance.conditionalSetValueGe(probeRadius, -1.f);

    // Initialize the patch.
    patch.initializePositive(grid.getPatchLowerCorner(*i), distance.grid);
    // Loop over the balls that are close to the patch and clip by the
    // probes that are positioned on each ball.
    clipByProbes(&patch, patchBall, closeToPatch, influencing);

    // Assign volume to the appropriate balls.
    unsigned char mask;
    const unsigned* id;
    if (patch.hasActive()) {
      for (std::size_t j = 0; j != patch.numActive; ++j) {
        mask = patch.activeMasks[j];
        id = reinterpret_cast<const unsigned*>
             (&distance.identifiers[patch.activeIndices[j]]);
        for (std::size_t k = 0; k != VectorSize; ++k, mask >>= 1) {
          if (mask & 1) {
            (*volumes)[id[k]] += voxelVolume;
          }
        }
      }
    }
  }
}


// Compute the volume of the solvent-excluded cavities and avoid storing any
// level-set function on a grid. Only a patch at a time will be used.
// Solvent probes will be placed by distributing points on a sphere
// (hence the Pos in the name).
// Compute the volume using only the sign of the distance.
inline
float
solventExcludedCavitiesPos(const std::vector<geom::Ball<float, 3> >& balls,
                           const float probeRadius,
                           const float targetGridSpacing)
{
  const std::size_t D = 3;
  typedef GridGeometry<D, PatchExtent, float> Grid;
  typedef Grid::BBox BBox;
  typedef geom::Ball<float, D> Ball;

  //
  // Define the grid geometry for computing the solvent-excluded cavities.
  //
  // Place a bounding box around the balls comprising the molecule.
  BBox targetDomain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // Define the grid geometry.
  const Grid grid(targetDomain, targetGridSpacing);

  // A ball influences a patch if it is within a distance of 2 * probeRadius.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // Expand the balls.
    std::vector<Ball> offsetBalls(balls);
    const float offset = 2 * probeRadius;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += offset;
    }
    patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

  // Compute the volume of the solvent-excluded cavities.
  return solventExcludedCavitiesPos(grid, balls, probeRadius, dependencies);
}


inline
void
solventExcludedCavitiesPos(const std::vector<geom::Ball<float, 3> >& balls,
                           const float probeRadius,
                           const float targetGridSpacing,
                           std::vector<float>* volumes)
{
  const std::size_t D = 3;
  typedef GridGeometry<D, PatchExtent, float> Grid;
  typedef Grid::BBox BBox;
  typedef geom::Ball<float, D> Ball;

  //
  // Define the grid geometry for computing the solvent-excluded cavities.
  //
  // Place a bounding box around the balls comprising the molecule.
  BBox targetDomain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // Define the grid geometry.
  const Grid grid(targetDomain, targetGridSpacing);

  // A ball influences a patch if it is within a distance of 2 * probeRadius.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // Expand the balls.
    std::vector<Ball> offsetBalls(balls);
    const float offset = 2 * probeRadius;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += offset;
    }
    patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

  // Compute the volume of the solvent-excluded cavities.
  solventExcludedCavitiesPos(grid, balls, probeRadius, dependencies, volumes);
}


} // namespace levelSet
}

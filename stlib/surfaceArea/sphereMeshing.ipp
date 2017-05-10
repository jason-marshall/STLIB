// -*- C++ -*-

#if !defined(__surfaceArea_sphereMeshing_ipp__)
#error This file is an implementation detail of sphereMeshing.
#endif

namespace stlib
{
namespace surfaceArea
{


// Distribute points on spheres. The point density and the sphere radii
// are specified.
template<typename Float>
inline
void
meshSpheres(std::vector<std::vector<std::array<Float, 3> > >* meshes,
            const Float pointDensity, const std::vector<Float>& radii)
{
  // A Cartesian point.
  typedef std::array<Float, 3> Point;

  meshes->resize(radii.size());
  std::vector<Point> points;
  // For each atom.
  for (std::size_t i = 0; i != radii.size(); ++i) {
    const Float area = 4. * numerical::Constants<Float>::Pi() *
                         radii[i] * radii[i];
    const std::size_t numPoints =
      static_cast<std::size_t>(area * pointDensity);
    // Place points on a unit sphere.
    distributePointsOnSphereWithGoldenSectionSpiral<Point>
    (numPoints, std::back_inserter(points));
    // Scale by the radius.
    points *= radii[i];
    // Copy.
    (*meshes)[i] = points;
    points.clear();
  }
}


// The input is a set of points along with the number of active points.
// The active points are in the range [0..numActive). The inactive points
// are in the range [numActive..points->size()). The active points are
// clipped by a sphere (defined by the specified center and radius).
// Clipped points are moved into the inactive range. The number of active
// points after clipping is returned.
template<typename Float>
inline
std::size_t
clip(std::vector<std::array<Float, 3> >* points,
     std::size_t numActive, const std::array<Float, 3>& center,
     const Float radius)
{
  const Float squaredRadius = radius * radius;
  std::size_t i = 0;
  while (i != numActive) {
    // If the point is inside the sphere.
    if (ext::squaredDistance((*points)[i], center) < squaredRadius) {
      // Move it to the inactive range.
      --numActive;
      std::swap((*points)[i], (*points)[numActive]);
    }
    else {
      /// Move to the next point.
      ++i;
    }
  }
  // Return the number of active points after clipping.
  return numActive;
}


// Given a set of N atoms, generate a set of points that represents the
// surface of the union of them. The points are generated on a per-atom
// basis, that is the output is a list of meshes, one for each atom.
// Parameters:
// - clippedSurfaces: The output meshes.
// - centers: Vector of N atom centers.
// - atomIndices: Vector of N atom identifiers.
// - referenceRadii: Vector of R radii, where R is the number at types.
// - referenceMeshes: Vector of R meshes of the reference atoms (centered
//   at the origin).
template<typename Float>
inline
void
meshBoundaryOfUnion
(std::vector<std::vector<std::array<Float, 3> > >* clippedSurfaces,
 const std::vector<std::array<Float, 3> >& centers,
 const std::vector<std::size_t> atomIndices,
 const std::vector<Float> referenceRadii,
 const std::vector<std::vector<std::array<Float, 3> > >&
 referenceMeshes)
{
  // Initialize.
  clippedSurfaces->resize(centers.size());
  // For each atom in the set.
  for (std::size_t i = 0; i != clippedSurfaces->size(); ++i) {
    // The atom type index for the atom being meshed.
    const std::size_t a = atomIndices[i];
    // Start with the reference mesh.
    (*clippedSurfaces)[i] = referenceMeshes[a];
    // Reference for a more convenient name.
    std::vector<std::array<Float, 3> >& mesh = (*clippedSurfaces)[i];
    // Translate to the specified center.
    mesh += centers[i];
    // For each of the other spheres.
    for (std::size_t j = 0; j != clippedSurfaces->size(); ++j) {
      if (i == j) {
        continue;
      }
      // The atom type index for the clipping atom.
      const std::size_t b = atomIndices[j];
      // If the two spheres intersect.
      if (ext::squaredDistance(centers[i], centers[j]) <
          (referenceRadii[a] + referenceRadii[b]) *
          (referenceRadii[a] + referenceRadii[b])) {
        // Clip the mesh.
        const std::size_t numActive =
          clip(&mesh, mesh.size(), centers[j], referenceRadii[b]);
        // Discard the clipped points.
        mesh.erase(mesh.begin() + numActive, mesh.end());
      }
    }
  }
}


template<typename Float>
inline
void
pairwiseSasa(std::vector<std::vector<std::array<Float, 3> > >*
             expandedAtomMeshes,
             const Float probeRadius, const Float pointDensity,
             const std::vector<Float>& atomRadii)
{
  // Expand the atomic radii by the probe radius.
  std::vector<Float> atomExpandedRadii(atomRadii);
  atomExpandedRadii += probeRadius;

  // Generate (point) meshes for each of the atom types.
  meshSpheres(expandedAtomMeshes, pointDensity, atomExpandedRadii);

  // CONTINUE
  //const Number _maxBallRadius = max(_atomExpandedRadii);
}

} // namespace surfaceArea
}

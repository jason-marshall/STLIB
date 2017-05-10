// -*- C++ -*-

#include "stlib/surfaceArea/sphereMeshing.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef float Number;
  // A Cartesian point.
  typedef std::array<Number, 3> Point;

  // pairwiseSasa
  {
    const Number probeRadius = 1.4;
    const Number pointDensity = 10.;
    std::vector<Number> atomRadii;
    atomRadii.push_back(2.);
    atomRadii.push_back(3.);
    std::vector<std::vector<Point> > meshes;
    surfaceArea::pairwiseSasa(&meshes, probeRadius, pointDensity,
                              atomRadii);
    // The correct number of meshes.
    assert(meshes.size() == atomRadii.size());
    // For each mesh.
    for (std::size_t i = 0; i != meshes.size(); ++i) {
      const Number radius = atomRadii[i] + probeRadius;
      const Number area = 4. * numerical::Constants<Number>::Pi() *
                          radius * radius;
      const std::vector<Point>& mesh = meshes[i];
      // Check the density of points.
      assert(std::abs(mesh.size() / area - pointDensity) <
             2. / area);
      for (std::size_t j = 0; j != mesh.size(); ++j) {
        // Check that the points have the correct radius.
        assert(std::abs(ext::magnitude(mesh[j]) - radius) < 10. *
               std::numeric_limits<Number>::epsilon());
      }
    }
  }

  // clip
  {
    std::vector<Point> points;
    Point center = {{1, 2, 3}};
    Number radius = 4;

    // Inside.
    points.push_back(center);
    assert(surfaceArea::clip(&points, 1, center, radius) == 0);

    // Inside, close to the sphere.
    points[0][0] = center[0] + radius *
                   (1. - std::numeric_limits<Number>::epsilon());
    assert(surfaceArea::clip(&points, 1, center, radius) == 0);

    // Outside, close to the sphere.
    points[0][0] = center[0] + radius *
                   (1. + std::numeric_limits<Number>::epsilon());
    assert(surfaceArea::clip(&points, 1, center, radius) == 1);

    // Inside, outside.
    points.resize(2);
    points[0] = center;
    const Point outside = {{10, 0, 0}};
    points[1] = outside;
    assert(surfaceArea::clip(&points, 2, center, radius) == 1);
    assert(points[0] == outside);
    assert(points[1] == center);

    // Outside, inside.
    points[0] = outside;
    points[1] = center;
    assert(surfaceArea::clip(&points, 2, center, radius) == 1);
    assert(points[0] == outside);
    assert(points[1] == center);
  }

  // meshBoundaryOfUnion
  {
    // Single atom of unit radius at the origin.
    std::vector<Point> centers;
    centers.push_back(Point{{0, 0, 0}});
    std::vector<std::size_t> atomIndices;
    atomIndices.push_back(0);
    std::vector<Number> referenceRadii;
    referenceRadii.push_back(1);
    std::vector<std::vector<Point> > referenceMeshes;
    const Number pointDensity = 10.;
    surfaceArea::meshSpheres(&referenceMeshes, pointDensity,
                             referenceRadii);
    std::vector<std::vector<Point> > clippedSurfaces;
    // Generate the surface of a single atom.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(clippedSurfaces[0].size() == referenceMeshes[0].size());

    // Add an atom of the same type that does not intersect the first.
    centers.push_back(Point{{3, 0, 0}});
    atomIndices.push_back(0);
    // Generate the surface of the two disjoint atoms.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(clippedSurfaces[0].size() == referenceMeshes[0].size());
    assert(clippedSurfaces[1].size() == referenceMeshes[0].size());

    // Move the second atom closer so that it intersects the first.
    centers[1][0] = 1;
    // Generate the surface of the two intersecting atoms.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(0 < clippedSurfaces[0].size() &&
           clippedSurfaces[0].size() < referenceMeshes[0].size());
    assert(0 < clippedSurfaces[1].size() &&
           clippedSurfaces[1].size() < referenceMeshes[0].size());

    // Change the type of the second atom to one with a radius of 2.
    // Move the second atom so that the two do not intersect.
    atomIndices[1] = 1;
    referenceRadii.push_back(2);
    surfaceArea::meshSpheres(&referenceMeshes, pointDensity,
                             referenceRadii);
    centers[1][0] = 4;
    // Generate the surface of the two disjoint atoms.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(clippedSurfaces[0].size() == referenceMeshes[0].size());
    assert(clippedSurfaces[1].size() == referenceMeshes[1].size());


    // Move the second atom closer so that it intersects the first.
    centers[1][0] = 2;
    // Generate the surface of the two intersecting atoms.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(0 < clippedSurfaces[0].size() &&
           clippedSurfaces[0].size() < referenceMeshes[0].size());
    assert(0 < clippedSurfaces[1].size() &&
           clippedSurfaces[1].size() < referenceMeshes[1].size());

    // Move the second atom closer so that it covers the first.
    centers[1][0] = 0;
    // Generate the surface of the two intersecting atoms.
    surfaceArea::meshBoundaryOfUnion(&clippedSurfaces, centers, atomIndices,
                                     referenceRadii, referenceMeshes);
    assert(clippedSurfaces.size() == centers.size());
    assert(clippedSurfaces[0].size() == 0);
    assert(clippedSurfaces[1].size() == referenceMeshes[1].size());
  }

  return 0;
}

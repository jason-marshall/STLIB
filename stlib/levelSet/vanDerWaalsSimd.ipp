// -*- C++ -*-

#if !defined(__levelSet_vanDerWaalsSimd_ipp__)
#error This file is an implementation detail of vanDerWaalsSimd.
#endif

namespace stlib
{
namespace levelSet
{


float
vanDerWaalsSimd
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies)
{
  typedef container::SimpleMultiIndexExtentsIterator<3> Iterator;
  assert(dependencies.getNumberOfArrays() == ext::product(grid.gridExtents));

  PatchActive patch(grid.spacing);
  std::size_t negativeCount = 0;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.gridExtents);
  for (Iterator i = Iterator::begin(grid.gridExtents); i != end; ++i) {
    // Initialize the patch.
    patch.initialize(grid.getPatchLowerCorner(*i), true);
    // Clip by each of the influencing balls.
    const std::size_t index = grid.arrayIndex(*i);
    for (std::size_t j = 0; j != dependencies.size(index); ++j) {
      patch.clip(balls[dependencies(index, j)]);
    }
    // Add the number of negative grid points in this patch.
    negativeCount += PatchActive::NumPoints - patch.numActivePoints();
  }

  // Return the total volume.
  return negativeCount * grid.spacing * grid.spacing * grid.spacing;
}


float
vanDerWaalsSimd(const std::vector<geom::Ball<float, 3> >& balls,
                float targetGridSpacing)
{
  const std::size_t D = 3;
  typedef GridGeometry<D, PatchExtent, float> Grid;
  typedef Grid::BBox BBox;

  //
  // Define the grid geometry for computing the van der Waals volume.
  //
  // Place a bounding box around the balls comprising the molecule.
  BBox targetDomain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // Define the grid geometry.
  const Grid grid(targetDomain, targetGridSpacing);

  // Compute the patch dependencies.
  container::StaticArrayOfArrays<unsigned> dependencies;
  patchDependencies(grid, balls.begin(), balls.end(), &dependencies);

  // Compute the van der Waals volume.
  return vanDerWaalsSimd(grid, balls, dependencies);
}


} // namespace levelSet
}

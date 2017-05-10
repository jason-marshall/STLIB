// -*- C++ -*-

#if !defined(__levelSet_vanDerWaals_ipp__)
#error This file is an implementation detail of vanDerWaals.
#endif

namespace stlib
{
namespace levelSet
{


template<typename _T, std::size_t _D>
inline
std::pair<_T, _T>
vanDerWaals(const std::vector<geom::Ball<_T, _D> >& balls,
            const _T targetGridSpacing)
{
  const std::size_t N = 8;

  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::BBox BBox;

  // Place a bounding box around the balls.
  BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // Expand by the diagonal length of a voxel so that we capture the
  // zero iso-surface.
  offset(&domain, targetGridSpacing * std::sqrt(_T(_D)));
  // Make the grid.
  Grid grid(domain, targetGridSpacing);

  // Calculate the power distance to the union of the balls.
  negativePowerDistance(&grid, balls);

  // Compute the content (volume) and boundary (surface area).
  _T content, boundary;
  levelSet::contentAndBoundary(grid, &content, &boundary);
  return std::make_pair(content, boundary);
}


} // namespace levelSet
}

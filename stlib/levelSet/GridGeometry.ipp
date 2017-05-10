// -*- C++ -*-

#if !defined(__levelSet_GridGeometry_ipp__)
#error This file is an implementation detail of GridGeometry.
#endif

namespace stlib
{
namespace levelSet
{


// Construct from the Cartesian domain and the suggested grid patch spacing.
template<std::size_t _D, std::size_t N, typename _R>
inline
GridGeometry<_D, N, _R>::
GridGeometry(const BBox& domain, const _R targetSpacing) :
  gridExtents(calculateExtents(domain, targetSpacing)),
  lowerCorner(domain.lower),
  // Choose the maximum over the dimensions in order to cover the domain.
  spacing(ext::max((domain.upper - domain.lower) /
                   ext::convert_array<_R>(gridExtents * N - Index(1)))),
  _strides(computeStrides(gridExtents))
{
  // Ensure that the patch extent is a power of 2.
  // http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
  static_assert(N && !(N & (N - 1)), "Patch extent must be a power of 2.");
  assert(spacing <= targetSpacing);
}


// Construct from the Cartesian domain and the grid extents.
template<std::size_t _D, std::size_t N, typename _R>
inline
GridGeometry<_D, N, _R>::
GridGeometry(const BBox& domain, const IndexList& extents) :
  gridExtents(extents),
  // Don't allocate memory for the patch arrays until the grid is refined.
  lowerCorner(domain.lower),
  // Choose the maximum over the dimensions in order to cover the domain.
  spacing(ext::max((domain.upper - domain.lower) /
                   ext::convert_array<_R>(extents * N - Index(1)))),
  _strides(computeStrides(extents))
{
  // Ensure that the patch extent is a power of 2.
  // http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
  static_assert(N && !(N & (N - 1)), "Patch extent must be a power of 2.");
}


template<std::size_t _D, std::size_t N, typename _R>
inline
typename GridGeometry<_D, N, _R>::IndexList
GridGeometry<_D, N, _R>::
calculateExtents(const BBox& domain, const _R targetSpacing)
{
  // Number of cells = ext * N. Note that we don't count the layer of
  // boundary vertices along the upper sides.
  IndexList ext;
  for (std::size_t i = 0; i != ext.size(); ++i) {
    // Include a fudge factor for the length.
    const _R length = (domain.upper[i] - domain.lower[i]) *
                      (1 + std::numeric_limits<_R>::epsilon());
    // length = dx * (ext * N - 1)
    // ext = (length / dx + 1) / N
    ext[i] = std::size_t(std::ceil((length / targetSpacing + 1)
                                   / N));
    assert(ext[i] != 0);
    // Ensure that there are at least two grid points.
    if (N == 1 && ext[i] == 1) {
      ext[i] = 2;
    }
  }
  return ext;
}


// Return the Cartesian domain spanned by the grid.
template<std::size_t _D, std::size_t N, typename _R>
inline
typename GridGeometry<_D, N, _R>::BBox
GridGeometry<_D, N, _R>::
domain() const
{
  BBox domain = {lowerCorner,
                 lowerCorner + stlib::ext::convert_array<_R>
                 (gridExtents * N - std::size_t(1)) * spacing};
  return domain;
}


// Return the domain for the specified vertex patch.
template<std::size_t _D, std::size_t N, typename _R>
inline
typename GridGeometry<_D, N, _R>::BBox
GridGeometry<_D, N, _R>::
getVertexPatchDomain(const IndexList& i) const
{
  const Point lower = lowerCorner +
    getVoxelPatchLength() * stlib::ext::convert_array<_R>(i);
  const BBox domain = {lower, lower + spacing * (N - 1)};
  return domain;
}


// Return the Cartesian position of the specified vertex.
template<std::size_t _D, std::size_t N, typename _R>
inline
typename GridGeometry<_D, N, _R>::Point
GridGeometry<_D, N, _R>::
indexToLocation(const IndexList& patch,
                const IndexList& index) const
{
  Point x;
  for (std::size_t d = 0; d != Dimension; ++d) {
    x[d] = lowerCorner[d] + patch[d] * getVoxelPatchLength() +
           index[d] * spacing;
  }
  return x;
}


// Report the specified set of grid points as patch/grid multi-index pairs.
template<std::size_t _D, std::size_t N, typename _R>
template<typename _OutputIterator>
inline
void
GridGeometry<_D, N, _R>::
report(const IndexList& patch, const Range& range,
       _OutputIterator neighbors) const
{
  typedef IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;
  std::pair<IndexList, IndexList> dual;
  dual.first = patch;
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    dual.second = *i;
    *neighbors++ = dual;
  }
}


// Determine the objects whose bounding boxes overlap each patch.
template<std::size_t _D, std::size_t N, typename _R, typename _InputIterator>
inline
void
patchDependencies(const GridGeometry<_D, N, _R>& grid, _InputIterator begin,
                  _InputIterator end,
                  container::StaticArrayOfArrays<unsigned>* dependencies)
{
  typedef GridGeometry<_D, N, _R> GridGeometry;
  typedef typename GridGeometry::Index Index;
  typedef typename GridGeometry::IndexList IndexList;
  typedef container::SimpleMultiIndexRange<_D> Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;
  typedef std::pair<std::size_t, std::size_t> PatchObjectPair;

  // The inverse length of a voxel patch.
  const _R inverseLength = 1. / grid.getVoxelPatchLength();

  geom::BBox<_R, _D> box;
  IndexList lower, upper;
  Range range;
  PatchObjectPair p;
  // The patch/object dependency pairs.
  std::vector<PatchObjectPair> pairs;
  // For each object, record the patches that depend on it.
  for (std::size_t objectIndex = 0; begin != end; ++begin, ++objectIndex) {
    // Make a bounding box around the object.
    box = geom::specificBBox<geom::BBox<_R, _D> >(*begin);
    //
    // Convert the Cartesian coordinates to index coordinates.
    //
    box.lower -= grid.lowerCorner;
    box.lower *= inverseLength;
    for (std::size_t i = 0; i != _D; ++i) {
      // Round down. Use only voxels in the grid range.
      lower[i] = Index(std::max(std::floor(box.lower[i]), _R(0.)));
    }
    box.upper -= grid.lowerCorner;
    box.upper *= inverseLength;
    for (std::size_t i = 0; i != _D; ++i) {
      // Round up for an open upper bound. Use only voxels in the grid range.
      upper[i] = Index(std::max(std::min(std::ceil(box.upper[i]),
                                         _R(grid.gridExtents[i])),
                                _R(lower[i])));
    }
    range.extents = upper - lower;
    range.bases = lower;

    // Loop over the range.
    p.second = objectIndex;
    const Iterator end = Iterator::end(range);
    for (Iterator i = Iterator::begin(range); i != end; ++i) {
      p.first = grid.arrayIndex(*i);
      pairs.push_back(p);
    }
  }

  // Determine the number of dependencies for each patch.
  std::vector<std::size_t> sizes(ext::product(grid.gridExtents), 0);
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    ++sizes[pairs[i].first];
  }

  // Allocate memory for the dependencies array.
  dependencies->rebuild(sizes.begin(), sizes.end());

  // Record the dependencies.
  std::fill(sizes.begin(), sizes.end(), 0);
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    const std::size_t j = pairs[i].first;
    (*dependencies)(j, sizes[j]++) = pairs[i].second;
  }
}


template<std::size_t _D, std::size_t N, typename _R, typename _OutputIterator>
inline
void
getIntersectingPatches(const GridGeometry<_D, N, _R>& grid,
                       geom::BBox<_R, _D> box, _OutputIterator indices)
{
  typedef std::array<std::size_t, _D> IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Convert the box to index coordinates.
  const _R inverseGridGeometrySpacing = 1. / grid.spacing;
  box.lower -= grid.lowerCorner;
  box.lower *= inverseGridGeometrySpacing;
  box.upper -= grid.lowerCorner;
  box.upper *= inverseGridGeometrySpacing;

  // Determine the index range.
  // Closed lower and open upper bounds.
  IndexList lower, upper;
  for (std::size_t i = 0; i != _D; ++i) {
    // Lower.
    if (box.lower[i] < 0) {
      lower[i] = 0;
    }
    else if (box.lower[i] >= grid.gridExtents[i]) {
      lower[i] = grid.gridExtents[i];
    }
    else {
      lower[i] = std::size_t(box.lower[i]);
    }
    // Upper.
    if (box.upper[i] < 0) {
      upper[i] = 0;
    }
    else if (box.upper[i] >= grid.gridExtents[i]) {
      upper[i] = grid.gridExtents[i];
    }
    else {
      upper[i] = std::size_t(box.upper[i]) + 1;
    }
  }
  // Construct from the extents and bases.
  const container::SimpleMultiIndexRange<_D> range = {upper - lower, lower};
  // Loop over the index range and report the array indices.
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    *indices++ = grid.arrayIndex(*i);
  }
}


template<std::size_t _D, std::size_t N, typename _R>
inline
void
printInfo(const GridGeometry<_D, N, _R>& grid, std::ostream& out)
{
  out << "Extents = " << grid.gridExtents << '\n'
      << "Domain = " << grid.domain() << '\n'
      << "Spacing = " << grid.spacing << '\n'
      << "Patch extent = " << N << '\n';
}


} // namespace levelSet
}

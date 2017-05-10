// -*- C++ -*-

#if !defined(__levelSet_GridVirtual_ipp__)
#error This file is an implementation detail of GridVirtual.
#endif

namespace stlib
{
namespace levelSet
{


// Construct from the Cartesian domain and the suggested grid patch spacing.
template<typename _T, std::size_t _D, std::size_t N>
inline
GridVirtual<_T, _D, N>::
GridVirtual(const BBox& domain, const _T targetSpacing) :
  extents(calculateExtents(domain, targetSpacing)),
  // Don't allocate memory for the patch arrays until the grid is refined.
  lowerCorner(domain.lower),
  // Choose the maximum over the dimensions in order to cover the domain.
  spacing(ext::max((domain.upper - domain.lower) /
                   ext::convert_array<_T>(extents * (N - 1)))),
  _strides(calculateStrides(extents))
{
  static_assert(N > 1, "Bad patch extent.");
  assert(spacing <= targetSpacing);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::IndexList
GridVirtual<_T, _D, N>::
calculateExtents(const BBox& domain, const _T targetSpacing)
{
  // Number of cells = ext * (N - 1).
  IndexList e;
  for (std::size_t i = 0; i != e.size(); ++i) {
    // Include a fudge factor for the length.
    const _T length = (domain.upper[i] - domain.lower[i]) *
                      (1 + std::numeric_limits<_T>::epsilon());
    // length = dx * e * (N - 1)
    // e = length / (dx * (N - 1))
    e[i] = std::size_t(std::ceil(length / (targetSpacing * (N - 1))));
    assert(e[i] != 0);
  }
  return e;
}


template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::IndexList
GridVirtual<_T, _D, N>::
calculateStrides(const IndexList& ext)
{
  IndexList strides;
  strides[0] = 1;
  for (std::size_t i = 1; i != Dimension; ++i) {
    strides[i] = strides[i - 1] * ext[i - 1];
  }
  return strides;
}


// Return the Cartesian domain spanned by the grid.
template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::BBox
GridVirtual<_T, _D, N>::
domain() const
{
  BBox domain = {lowerCorner,
                 lowerCorner + extents* (N - 1)* spacing
                };
  return domain;
}


// Return the domain for the specified patch.
template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::BBox
GridVirtual<_T, _D, N>::
getPatchDomain(const IndexList& i) const
{
  const Point lower = lowerCorner + getPatchLength() * i;
  const BBox domain = {lower, lower + getPatchLength()};
  return domain;
}


// Return the lower corner of the Cartesian domain for the specified patch.
template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::Point
GridVirtual<_T, _D, N>::
getPatchLowerCorner(const IndexList& i) const
{
  return lowerCorner + getPatchLength() * stlib::ext::convert_array<_T>(i);
}


// Return the Cartesian position of the specified vertex.
template<typename _T, std::size_t _D, std::size_t N>
inline
typename GridVirtual<_T, _D, N>::Point
GridVirtual<_T, _D, N>::
indexToLocation(const IndexList& patch,
                const IndexList& index) const
{
  Point x;
  for (std::size_t d = 0; d != Dimension; ++d) {
    x[d] = lowerCorner[d] + patch[d] * getPatchLength() +
           index[d] * spacing;
  }
  return x;
}


// Determine the objects whose bounding boxes overlap each patch.
template<typename _T, std::size_t _D, std::size_t N, typename _InputIterator>
inline
void
patchDependencies(const GridVirtual<_T, _D, N>& grid, _InputIterator begin,
                  _InputIterator end,
                  container::StaticArrayOfArrays<unsigned>* dependencies)
{
  typedef GridVirtual<_T, _D, N> GridVirtual;
  typedef typename GridVirtual::IndexList IndexList;
  typedef typename IndexList::value_type Index;
  typedef container::SimpleMultiIndexRange<_D> Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;
  typedef std::pair<std::size_t, std::size_t> PatchObjectPair;

  // The inverse length of a patch.
  const _T inverseLength = 1. / grid.getPatchLength();

  geom::BBox<_T, _D> box;
  IndexList lower, upper;
  Range range;
  PatchObjectPair p;
  // The patch/object dependency pairs.
  std::vector<PatchObjectPair> pairs;
  // For each object, record the patches that depend on it.
  for (std::size_t objectIndex = 0; begin != end; ++begin, ++objectIndex) {
    // Make a bounding box around the object.
    box = geom::specificBBox<geom::BBox<_T, _D> >(*begin);
    //
    // Convert the Cartesian coordinates to index coordinates.
    //
    box.lower -= grid.lowerCorner;
    box.lower *= inverseLength;
    for (std::size_t i = 0; i != _D; ++i) {
      // Round down. Use only voxels in the grid range.
      lower[i] = Index(std::max(std::floor(box.lower[i]), _T(0.)));
    }
    box.upper -= grid.lowerCorner;
    box.upper *= inverseLength;
    for (std::size_t i = 0; i != _D; ++i) {
      // Round up for an open upper bound. Use only voxels in the grid range.
      upper[i] = Index(std::max(std::min(std::ceil(box.upper[i]),
                                         _T(grid.extents[i])),
                                _T(lower[i])));
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
  std::vector<std::size_t> sizes(ext::product(grid.extents), 0);
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


template<typename _T, std::size_t _D, std::size_t N,
         typename _OutputIterator>
inline
void
getIntersectingPatches(const GridVirtual<_T, _D, N>& grid,
                       geom::BBox<_T, _D> box, _OutputIterator indices)
{
  typedef std::array<std::size_t, _D> IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Convert the box to index coordinates.
  const _T inverseGridSpacing = 1. / grid.spacing;
  box.lower -= grid.lowerCorner;
  box.lower *= inverseGridSpacing;
  box.upper -= grid.lowerCorner;
  box.upper *= inverseGridSpacing;

  // Determine the index range.
  // Closed lower and open upper bounds.
  IndexList lower, upper;
  for (std::size_t i = 0; i != _D; ++i) {
    // Lower.
    if (box.lower[i] < 0) {
      lower[i] = 0;
    }
    else if (box.lower[i] >= grid.extents[i]) {
      lower[i] = grid.extents[i];
    }
    else {
      lower[i] = std::size_t(box.lower[i]);
    }
    // Upper.
    if (box.upper[i] < 0) {
      upper[i] = 0;
    }
    else if (box.upper[i] >= grid.extents[i]) {
      upper[i] = grid.extents[i];
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


template<typename _T, std::size_t _D, std::size_t N>
inline
void
printInfo(const GridVirtual<_T, _D, N>& grid, std::ostream& out)
{
  out << "Extents = " << grid.extents << '\n'
      << "Lower corner = " << grid.lowerCorner << '\n'
      << "Spacing = " << grid.spacing << '\n'
      << "Domain = " << grid.domain() << '\n'
      << "Patch extent = " << N << '\n';
}


} // namespace levelSet
}

// -*- C++ -*-

#if !defined(__levelSet_components_ipp__)
#error This file is an implementation detail of components.
#endif

namespace stlib
{
namespace levelSet
{

// Label the component of which the specified grid point is a member.
template<typename _T, std::size_t _D, std::size_t N, typename _R, typename _I>
inline
void
labelComponent(const Grid<_T, _D, N, _R>& f, Grid<_I, _D, N, _R>* components,
               const typename Grid<_T, _D, N, _R>::IndexList& gridIndex,
               const typename Grid<_T, _D, N, _R>::IndexList& patchIndex,
               const _I identifier)
{
  typedef Grid<_T, _D, N, _R> Grid;
  typedef typename Grid::DualIndices DualIndices;

  std::deque<DualIndices> queue;
  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Start with the one known member.
  DualIndices di;
  di.first = gridIndex;
  di.second = patchIndex;
  assert(f(di) <= 0);
  (*components)(di) = identifier;
  queue.push_back(di);
  // Find the rest of the component.
  while (! queue.empty()) {
    di = queue.front();
    queue.pop_front();
    // Examine the set of all neighboring grid points.
    neighbors.clear();
    f.allNeighbors(di, insertIterator);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      const DualIndices& n = neighbors[i];
      _I& c = (*components)(n);
      if (f(n) <= 0 && c != identifier) {
#ifdef STLIB_DEBUG
        // We should not encounter a previously labeled component.
        assert(c > identifier);
#endif
        queue.push_back(n);
        c = identifier;
      }
    }
  }
}


// Identify the connected components of the level set function.
/*
  The grid points that are not part of a component are set to
  \c std::numeric_limits<_I>::max().
*/
template<typename _T, std::size_t _D, std::size_t N, typename _R, typename _I>
inline
_I
labelComponents(const Grid<_T, _D, N, _R>& f,
                Grid<_I, _D, N, _R>* components)
{
  typedef Grid<_T, _D, N, _R> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::VertexPatch FunctionPatch;
  typedef typename levelSet::Grid<_I, _D, N, _R>::VertexPatch ComponentPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Although we will refine the components grid, it must have the same
  // extents as the level set grid.
  assert(f.extents() == components->extents());

  // Refine the same patches as in the level set for the function.
  {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i != f.size(); ++i) {
      if (f[i].isRefined()) {
        indices.push_back(i);
      }
    }
    components->refine(indices);
  }
  // Initialize all grid points as not belonging to a component.
  for (std::size_t i = 0; i != components->size(); ++i) {
    ComponentPatch& patch = (*components)[i];
    if (patch.isRefined()) {
      std::fill(patch.begin(), patch.end(), std::numeric_limits<_I>::max());
    }
    else {
      patch.fillValue = std::numeric_limits<_I>::max();
    }
  }

  // Loop over the grid points, patch by patch.
  _I componentIndex = 0;
  const Iterator pEnd = Iterator::end(f.extents());
  for (Iterator p = Iterator::begin(f.extents()); p != pEnd; ++p) {
    const FunctionPatch& fp = f(*p);
    const ComponentPatch& cp = (*components)(*p);
    if (fp.isRefined()) {
      // Loop over the grid points in the patch.
      const Iterator iEnd = Iterator::end(fp.extents());
      for (Iterator i = Iterator::begin(fp.extents()); i != iEnd; ++i) {
        // If this is a part of a new component.
        if (fp(*i) <= 0 && cp(*i) > componentIndex) {
          labelComponent(f, components, *p, *i, componentIndex);
          ++componentIndex;
          assert(componentIndex != std::numeric_limits<_I>::max());
        }
      }
    }
    else {
      // If this is a part of a new component.
      if (fp.fillValue <= 0 && cp.fillValue > componentIndex) {
        labelComponent(f, components, *p, ext::filled_array<IndexList>(0),
                       componentIndex);
        ++componentIndex;
        assert(componentIndex != std::numeric_limits<_I>::max());
      }
    }
  }

  // Return the number of components.
  return componentIndex;
}


} // namespace levelSet
}

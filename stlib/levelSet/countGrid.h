// -*- C++ -*-

#if !defined(__levelSet_countGrid_h__)
#define __levelSet_countGrid_h__

#include "stlib/levelSet/count.h"
#include "stlib/levelSet/Grid.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetCountGrid Count Known Values in an AMR grid
 These functions count the number of known or unknown values in an AMR grid. */
//@{


//! Return true if the grid has any known values.
template<typename _T, std::size_t _D, std::size_t N>
inline
bool
hasKnown(const Grid<_T, _D, N>& grid)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;
  // Loop over the patches.
  for (std::size_t i = 0; i != grid.size(); ++i) {
    const VertexPatch& p = grid[i];
    if (p.isRefined()) {
      if (hasKnown(p.begin(), p.end())) {
        return true;
      }
    }
    else {
      if (p.fillValue == p.fillValue) {
        return true;
      }
    }
  }
  return false;
}


//! Return true if the grid has any unknown values.
template<typename _T, std::size_t _D, std::size_t N>
inline
bool
hasUnknown(const Grid<_T, _D, N>& grid)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;
  // Loop over the patches.
  for (std::size_t i = 0; i != grid.size(); ++i) {
    const VertexPatch& p = grid[i];
    if (p.isRefined()) {
      if (hasUnknown(p.begin(), p.end())) {
        return true;
      }
    }
    else {
      if (p.fillValue != p.fillValue) {
        return true;
      }
    }
  }
  return false;
}


//! Return true if the grid point has an unknown neighbor.
template<typename _T, std::size_t _D, std::size_t N>
inline
bool
hasUnknownAdjacentNeighbor(const Grid<_T, _D, N>& grid,
                           const typename Grid<_T, _D, N>::DualIndices& ij)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::DualIndices DualIndices;

  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Get the adjacent neighbors.
  grid.adjacentNeighbors(ij, insertIterator);
  // Check if any are unknown.
  for (std::size_t i = 0; i != neighbors.size(); ++i) {
    if (grid(neighbors[i]) != grid(neighbors[i])) {
      return true;
    }
  }
  return false;
}


//! Return true if the grid point has a negative adjacent neighbor.
template<typename _T, std::size_t _D, std::size_t N>
inline
bool
hasNegativeAdjacentNeighbor(const Grid<_T, _D, N>& grid,
                            const typename Grid<_T, _D, N>::DualIndices& ij)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::DualIndices DualIndices;

  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Get the adjacent neighbors.
  grid.adjacentNeighbors(ij, insertIterator);
  // Check if any are negative.
  for (std::size_t i = 0; i != neighbors.size(); ++i) {
    if (grid(neighbors[i]) < 0) {
      return true;
    }
  }
  return false;
}


//! Print information about the grid.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
printLevelSetInfo(const Grid<_T, _D, N>& grid, std::ostream& out)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;
  const _T Inf = std::numeric_limits<_T>::infinity();

  std::size_t size = 0;
  std::size_t nonNegative = 0;
  std::size_t negative = 0;
  std::size_t unknown = 0;
  std::size_t positiveFar = 0;
  std::size_t negativeFar = 0;
  _T x;

  // Loop over the refined patches.
  for (std::size_t i = 0; i != grid.size(); ++i) {
    const VertexPatch& p = grid[i];
    if (! p.isRefined()) {
      continue;
    }
    for (typename VertexPatch::const_iterator j = p.begin(); j != p.end();
         ++j) {
      x = *j;
      ++size;
      if (x >= 0) {
        ++nonNegative;
      }
      else if (x < 0) {
        ++negative;
      }
      else {
        ++unknown;
      }
      if (x == Inf) {
        ++positiveFar;
      }
      else if (x == -Inf) {
        ++negativeFar;
      }
    }

  }

  out << "Number of refined grid points = " << size << '\n'
      << "known/unknown = " << size - unknown << " / " << unknown << '\n'
      << "non-negative/negative = " << nonNegative << " / " << negative
      << '\n'
      << "positive far/negative far = " << positiveFar << " / " << negativeFar
      << '\n';
}


//@}


} // namespace levelSet
}

#endif

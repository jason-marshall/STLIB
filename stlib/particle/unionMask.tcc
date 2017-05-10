// -*- C++ -*-

#if !defined(__particle_unionMask_tcc__)
#error This file is an implementation detail of unionMask.
#endif

namespace stlib
{
namespace particle
{


template<typename _Order>
inline
UnionMask<_Order>::
UnionMask(const _Order& order) :
  NeighborsPerformance(),
  unionOfNeighbors(),
  neighborMasks(),
  _order(order)
{
}


template<typename _Order>
inline
std::size_t
UnionMask<_Order>::
numNeighbors() const
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != neighborMasks.numArrays(); ++i) {
    count += numNeighbors(i);
  }
  return count;
}


template<typename _Order>
inline
std::size_t
UnionMask<_Order>::
numNeighbors(const std::size_t i) const
{
#ifdef STLIB_DEBUG
  assert(i < neighborMasks.numArrays());
#endif
  std::size_t count = 0;
  for (std::size_t j = 0; j != neighborMasks.size(i); ++j) {
    count += numerical::popCount(neighborMasks(i, j));
  }
  return count;
}


template<typename _Order>
inline
bool
UnionMask<_Order>::
isNeighbor(const std::size_t particle, const std::size_t index) const
{
#ifdef STLIB_DEBUG
  assert(particle < _order.particles.size());
  assert(index < numPotentialNeighbors(particle));
#endif
  // Extract the appropriate bit.
  const std::size_t i = index / MaskDigits;
  const std::size_t j = index % MaskDigits;
  return (neighborMasks(particle, i) >> j) & 1;
}


template<typename _Order>
inline
void
UnionMask<_Order>::
positionsInUnion(const std::size_t cell, std::vector<Point>* positions) const
{
  // Get the positions.
  positions->resize(unionOfNeighbors.size(cell));
  for (std::size_t i = 0; i != positions->size(); ++i) {
    (*positions)[i] = neighborPosition(unionOfNeighbors(cell, i));
  }
  // Pad with NaN's.
  _pad(cell, positions);
}


template<typename _Order>
inline
void
UnionMask<_Order>::
positionsInUnion(const std::size_t cell, std::vector<std::size_t>* indices,
                 std::vector<Point>* positions) const
{
  // Get the indices and positions.
  indices->resize(unionOfNeighbors.size(cell));
  positions->resize(unionOfNeighbors.size(cell));
  for (std::size_t i = 0; i != positions->size(); ++i) {
    const Neighbor& neighbor = unionOfNeighbors(cell, i);
    (*indices)[i] = neighbor.particle;
    (*positions)[i] = _order.neighborPosition(neighbor);
  }
  // Pad with NaN's.
  _pad(cell, indices, positions);
}


template<typename _Order>
inline
void
UnionMask<_Order>::
_pad(const std::size_t cell, std::vector<Point>* positions) const
{
#ifdef STLIB_DEBUG
  assert(neighborMasks.size(_order.cellBegin(cell)) * MaskDigits >=
         positions->size());
#endif
  const std::size_t n =
    neighborMasks.size(_order.cellBegin(cell)) * MaskDigits -
    positions->size();
  positions->insert(positions->end(), n,
                    ext::filled_array<Point>
                    (std::numeric_limits<Float>::quiet_NaN()));
}


template<typename _Order>
inline
void
UnionMask<_Order>::
_pad(const std::size_t cell, std::vector<std::size_t>* indices,
     std::vector<Point>* positions) const
{
#ifdef STLIB_DEBUG
  assert(indices->size() == positions->size());
  assert(neighborMasks.size(_order.cellBegin(cell)) * MaskDigits >=
         positions->size());
#endif
  const std::size_t n =
    neighborMasks.size(_order.cellBegin(cell)) * MaskDigits -
    positions->size();
  indices->insert(indices->end(), n, std::size_t(0));
  positions->insert(positions->end(), n,
                    ext::filled_array<Point>
                    (std::numeric_limits<Float>::quiet_NaN()));
}


template<typename _Order>
inline
void
UnionMask<_Order>::
_positionsInUnion(const std::vector<Point>& cachedPositions,
                  const std::size_t cell, std::vector<Point>* positions) const
{
  // Get the positions.
  positions->resize(unionOfNeighbors.size(cell));
  for (std::size_t i = 0; i != positions->size(); ++i) {
    (*positions)[i] = _neighborPosition(cachedPositions,
                                        unionOfNeighbors(cell, i));
  }
  // Pad with NaN's.
  _pad(cell, positions);
}


template<typename _Order>
inline
void
UnionMask<_Order>::
_getUnionOfNeighbors(const std::vector<Point>& cachedPositions,
                     const std::size_t cell, std::vector<Neighbor>* neighbors) const
{
  // Cache the positions of the particles in the cell.
  std::vector<Point> centerPositions(_order.cellEnd(cell) -
                                     _order.cellBegin(cell));
  {
    std::size_t p = _order.cellBegin(cell);
    // CONTINUE: Use a good ordering to improve early acceptance.
    for (std::size_t i = 0; i != centerPositions.size(); ++i, ++p) {
      centerPositions[i] = cachedPositions[p];
    }
  }
  Point pos;
  neighbors->clear();
  // For each of the adjacent cells.
  for (std::size_t i = 0; i != _order.adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& neighborCell =
      _order.adjacentCells(cell, i);
    if (neighborCell.cell == cell) {
      // Add all of the particles in the center cell.
      const Neighbor end = _order.cellEnd(neighborCell);
      for (Neighbor neighbor = _order.cellBegin(neighborCell);
           neighbor.particle != end.particle; ++neighbor) {
        neighbors->push_back(neighbor);
      }
    }
    else {
      // Check the distance for the other cells.
      const Neighbor end = _order.cellEnd(neighborCell);
      for (Neighbor neighbor = _order.cellBegin(neighborCell);
           neighbor.particle != end.particle; ++neighbor) {
        pos = _order.neighborPosition(cachedPositions, neighbor);
        for (std::size_t j = 0; j != centerPositions.size(); ++j) {
          if (ext::squaredDistance(pos, centerPositions[j]) <=
              _order.squaredInteractionDistance()) {
            neighbors->push_back(neighbor);
            break;
          }
        }
      }
    }
  }
}


template<typename _Order>
inline
void
UnionMask<_Order>::
_findNeighbors(const std::size_t cellsBegin, const std::size_t cellsEnd)
{
  _timer.start();
  ++_neighborsCount;
  std::vector<Point> cachedPositions;
  _order.getPositions(&cachedPositions);
  // Dispatch for generic and specialized implementations.
#ifdef _OPENMP
  // Make vectors of packed arrays, one for each thread.
  std::vector<container::PackedArrayOfArrays<Neighbor> >
  unionOfNeighbors_(omp_get_max_threads());
  std::vector<container::PackedArrayOfArrays<Mask> >
  neighborMasks_(omp_get_max_threads());
  // Determine the range of particles for each thread.
  std::vector<std::size_t> delimiters;
  _partitionCells(cellsBegin, cellsEnd, &delimiters);
  #pragma omp parallel default(none) shared(cachedPositions, delimiters, unionOfNeighbors_, neighborMasks_)
  {
    const std::size_t i = omp_get_thread_num();
    _findNeighbors(cachedPositions, delimiters[i], delimiters[i + 1],
                   &unionOfNeighbors_[i], &neighborMasks_[i], Float(0),
                   std::integral_constant<std::size_t, Dimension>());
  }
  unionOfNeighbors.rebuild(unionOfNeighbors_);
  neighborMasks.rebuild(neighborMasks_);
#else
  _findNeighbors(cachedPositions, cellsBegin, cellsEnd, &unionOfNeighbors,
                 &neighborMasks, Float(0),
                 std::integral_constant<std::size_t, Dimension>());
#endif
  _timer.stop();
  _timeNeighbors += _timer.elapsed();
}


template<typename _Order>
template<typename _Float, typename _Dimension>
inline
void
UnionMask<_Order>::
_findNeighbors(const std::vector<Point>& cachedPositions,
               const std::size_t cellsBegin, const std::size_t cellsEnd,
               container::PackedArrayOfArrays<Neighbor>* unionOfNeighbors_,
               container::PackedArrayOfArrays<Mask>* neighborMasks_,
               _Float /*dummy*/, _Dimension /*dummy*/)
{
  // The union of neighbors for a cell.
  std::vector<Neighbor> neighbors;
  // The positions of the union of neighbors for a cell.
  std::vector<Point> unionPos;
  // The masks for a particle.
  std::vector<Mask> masks;

  unionOfNeighbors_->clear();
  neighborMasks_->clear();

  // Loop over the cells.
  for (std::size_t cell = cellsBegin; cell != cellsEnd; ++cell) {
    // Record the union of neighbors for this cell.
    _getUnionOfNeighbors(cachedPositions, cell, &neighbors);
    unionOfNeighbors_->pushArray(neighbors.begin(), neighbors.end());
    // Get the positions of the union of neighbors.
    unionPos.resize(neighbors.size());
    for (std::size_t i = 0; i != unionPos.size(); ++i) {
      unionPos[i] = _order.neighborPosition(cachedPositions, neighbors[i]);
    }
    // Pad the positions to be a multiple of the mask size.
    {
      const std::size_t n = (unionPos.size() + MaskDigits - 1) /
                            MaskDigits * MaskDigits - unionPos.size();
      unionPos.insert(unionPos.end(), n,
                      ext::filled_array<Point>
                      (std::numeric_limits<Float>::quiet_NaN()));
    }
    masks.resize(unionPos.size() / MaskDigits);

    // Loop over the particles in the cell.
    for (std::size_t i = _order.cellBegin(cell); i != _order.cellEnd(cell);
         ++i) {
      const Point pos = cachedPositions[i];
      std::size_t index = 0;
      // Loop over the masks.
      for (std::size_t j = 0; j != masks.size(); ++j) {
        Mask mask = 0;
        Mask bit = 1;
        for (std::size_t k = 0; k != MaskDigits; ++k, ++index, bit <<= 1) {
          const Float d = ext::squaredDistance(pos, unionPos[index]);
          if (d > 0 && d <= _order.squaredInteractionDistance()) {
            mask |= bit;
          }
        }
        masks[j] = mask;
      }
      neighborMasks_->pushArray(masks.begin(), masks.end());
    }
  }
}


#ifdef _OPENMP
template<typename _Order>
inline
void
UnionMask<_Order>::
_partitionCells(const std::size_t localBegin, const std::size_t localEnd,
                std::vector<std::size_t>* delimiters) const
{
  delimiters->clear();
  // The weights are the number of adjacent neighbors for each local cell.
  std::vector<std::size_t> weights(_order.numAdjacentNeighbors.begin() +
                                   localBegin,
                                   _order.numAdjacentNeighbors.begin() + localEnd);
  // Determine the range of cells for each thread.
  numerical::computePartitions(weights, omp_get_max_threads(),
                               std::back_inserter(*delimiters));
  for (std::size_t i = 0; i != delimiters->size(); ++i) {
    (*delimiters)[i] += localBegin;
  }
}
#endif


template<typename _Order>
inline
void
UnionMask<_Order>::
printMemoryUsageTable(std::ostream& out) const
{
  out << ",total,per particle\n"
      << "union of neighbors lists,"
      << unionOfNeighbors.memoryUsage() << ','
      << double(unionOfNeighbors.memoryUsage()) / _order.particles.size()
      << '\n'
      << "neighbors masks,"
      << neighborMasks.memoryUsage() << ','
      << double(neighborMasks.memoryUsage()) / _order.particles.size() << '\n';
}


template<typename _Order>
inline
void
UnionMask<_Order>::
printNeighborDensity(std::ostream& out) const
{
  // Neighbor density.
  const std::size_t nn = numNeighbors();
  out << "Neighbor density = " << double(nn) /
      (neighborMasks.size() * MaskDigits) << '\n';
  // Nonzero mask density.
  std::size_t active = 0;
  for (std::size_t i = 0; i != neighborMasks.size(); ++i) {
    active += bool(neighborMasks[i]);
  }
  out << "Nonzero mask dens. = " << double(active) / neighborMasks.size()
      << ", eff. dens. = " << double(nn) / (active * MaskDigits)
      << ", eff. ops. = " << double(nn) / (active)
      << ".\n";
  // Nonzero mask density.
  Mask mask;
  active = 0;
  for (std::size_t i = 0; i != neighborMasks.size(); ++i) {
    mask = neighborMasks[i];
    for (std::size_t j = 0; j != MaskDigits / 4; ++j) {
      active += bool(mask & 0xFF);
      mask >>= 4;
    }
  }
  out << "Nonzero nibble dens. = " << double(active) /
      (neighborMasks.size() * MaskDigits / 4)
      << ", eff. dens. = " << double(nn) / (active * 4)
      << ", eff. ops. = " << double(nn) / (active)
      << ".\n";
}


template<typename _Order>
inline
void
UnionMask<_Order>::
printInfo(std::ostream& out) const
{
  std::size_t numWithNeighbors = 0;
  for (std::size_t i = 0; i != _order.particles.size(); ++i) {
    if (numNeighbors(i) != 0) {
      ++numWithNeighbors;
    }
  }
  out << "Dimension = " << Dimension << '\n'
      << "Number of particles = " << _order.particles.size() << '\n'
      << "Number with neighbors = " << numWithNeighbors << '\n';
}


} // namespace particle
}

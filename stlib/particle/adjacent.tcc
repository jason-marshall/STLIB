// -*- C++ -*-

#if !defined(__particle_adjacent_tcc__)
#error This file is an implementation detail of adjacent.
#endif

namespace stlib
{
namespace particle
{


template<typename _Order>
inline
AdjacentMask<_Order>::
AdjacentMask(const _Order& order) :
  NeighborsPerformance(),
  neighborMasks(),
  _order(order)
{
}


template<typename _Order>
inline
std::size_t
AdjacentMask<_Order>::
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
AdjacentMask<_Order>::
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
AdjacentMask<_Order>::
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
AdjacentMask<_Order>::
positionsInAdjacent(const std::size_t cell, std::vector<Point>* positions)
const
{
  // Get the positions.
  _order.positionsInAdjacent(cell, positions);
  // Pad with NaN's.
  _pad(cell, positions);
}


template<typename _Order>
inline
void
AdjacentMask<_Order>::
positionsInAdjacent(std::size_t cell, std::vector<std::size_t>* indices,
                    std::vector<Point>* positions) const
{
  // Get the positions.
  _order.positionsInAdjacent(cell, indices, positions);
  // Pad with NaN's.
#ifdef STLIB_DEBUG
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
AdjacentMask<_Order>::
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
AdjacentMask<_Order>::
_positionsInAdjacent(const std::vector<Point>& cachedPositions,
                     const std::size_t cell, std::vector<Point>* positions) const
{
  // Get the positions.
  _order.positionsInAdjacent(cachedPositions, cell, positions);
  // Pad with NaN's.
  _pad(cell, positions);
}


template<typename _Order>
inline
void
AdjacentMask<_Order>::
_initializeNeighborMasks(const std::size_t cellsBegin,
                         const std::size_t cellsEnd)
{
  // CONTINUE: Only rebuild when _order has been repaired.

  // Calculate the number of masks for each of the particles. Set the initial
  // value to zero so that the bit masks for the particles that precede and
  // follow the local particles are empty.
  std::vector<std::size_t> sizes(_order.particles.size(), std::size_t(0));
  // For each local cell.
  for (std::size_t cell = cellsBegin; cell != cellsEnd; ++cell) {
    // The required number of masks for the given number of adjacent
    // neighbors.
    const std::size_t numAdj = _order.countAdjacentParticles(cell);
    const std::size_t numMasks = (numAdj + MaskDigits - 1) / MaskDigits;
    // For each particle in the cell.
    for (std::size_t i = _order.cellBegin(cell); i != _order.cellEnd(cell);
         ++i) {
      // The number of masks needed for the particle.
      sizes[i] = numMasks;
    }
  }
  // Make the packed array from the sizes.
  neighborMasks.rebuild(sizes.begin(), sizes.end());
  // Fill it with zero values.
  std::fill(neighborMasks.begin(), neighborMasks.end(), Mask(0));
}


template<typename _Order>
inline
void
AdjacentMask<_Order>::
_findNeighbors(const std::size_t cellsBegin, const std::size_t cellsEnd)
{
  _timer.start();
  ++_neighborsCount;
  _initializeNeighborMasks(cellsBegin, cellsEnd);
  std::vector<Point> cachedPositions;
  _order.getPositions(&cachedPositions);
  // Dispatch for generic and specialized implementations.
#ifdef _OPENMP
  // Determine the range of particles for each thread.
  std::vector<std::size_t> delimiters;
  _partitionCells(cellsBegin, cellsEnd, &delimiters);
  #pragma omp parallel default(none) shared(cachedPositions, delimiters)
  {
    _findNeighbors(cachedPositions, delimiters[omp_get_thread_num()],
                   delimiters[omp_get_thread_num() + 1], Float(0),
                   std::integral_constant<std::size_t, Dimension>());
  }
#else
  _findNeighbors(cachedPositions, cellsBegin, cellsEnd, Float(0),
                 std::integral_constant<std::size_t, Dimension>());
#endif
  _timer.stop();
  _timeNeighbors += _timer.elapsed();
}


template<typename _Order>
template<typename _Float, typename _Dimension>
inline
void
AdjacentMask<_Order>::
_findNeighbors(const std::vector<Point>& cachedPositions,
               const std::size_t cellsBegin, const std::size_t cellsEnd,
               _Float /*dummy*/, _Dimension /*dummy*/)
{
  // The positions of the potential neighbors for a cell.
  std::vector<Point> adjPos;
  // Loop over the cells.
  for (std::size_t cell = cellsBegin; cell != cellsEnd; ++cell) {
    // Get the positions of the adjacent neighbors.
    _positionsInAdjacent(cachedPositions, cell, &adjPos);
    // The index of the particle itself in the neighbor list.
    std::size_t selfNeighborIndex = _order.centerCellOffset(cell);
    // Loop over the particles in the cell.
    for (std::size_t i = _order.cellBegin(cell); i != _order.cellEnd(cell);
         ++i, ++selfNeighborIndex) {
#ifdef STLIB_DEBUG
      assert(neighborMasks.size(i) * MaskDigits == adjPos.size());
#endif
      const Point pos = _order.position(i);
      std::size_t index = 0;
      // Loop over the masks.
      for (std::size_t j = 0; j != neighborMasks.size(i); ++j) {
        Mask mask = 0;
        Mask bit = 1;
        for (std::size_t k = 0; k != MaskDigits; ++k, ++index, bit <<= 1) {
          if (ext::squaredDistance(pos, adjPos[index]) <=
              _order.squaredInteractionDistance()) {
            mask |= bit;
          }
        }
        neighborMasks(i, j) = mask;
      }

      // Mark the particle itself as not being a neighbor.
#ifdef STLIB_DEBUG
      assert(selfNeighborIndex < neighborMasks.size(i) * MaskDigits);
      assert(pos == adjPos[selfNeighborIndex]);
#endif
      neighborMasks(i, selfNeighborIndex / MaskDigits) &=
        ~(Mask(1) << (selfNeighborIndex % MaskDigits));
#ifdef STLIB_DEBUG
      assert(! isNeighbor(i, selfNeighborIndex));
#endif
    }
  }
}


// CONTINUE: Implement this.
#if 0
#ifdef __SSE__
template<typename _Order>
inline
void
AdjacentMask<_Order>::
_findNeighbors(const std::size_t begin, const std::size_t end,
               float /*dummy*/, std::integral_constant<std::size_t, 3> /*3D*/)
{
  // The index of the particle itself in the neighbor list.
  std::size_t selfNeighborIndex;
  // The positions of the potential neighbors for a cell.
  std::vector<Point> positions;
  // The shuffled coordinates.
  std::vector<float, simd::allocator<float> > shuffled;
  // Loop over the cells.
  for (std::size_t i = begin; i != end; /*No increment.*/) {
    // Determine the neighbor index of the particle itself so it can be
    // marked as not being a neighbor.
    selfNeighborIndex = _selfNeighborIndex(i);
    // Get the positions of the potential neighbors.
    _extractPotentialNeighborPositions(i, &positions);
    // Shuffle the coordinates so that we can use SSE operations.
    simd::aosToHybridSoa(positions, &shuffled);

    // Loop over the particles in the cell.
    const Code code = _order.codes()[i];
    for (; _order.codes()[i] == code; ++i, ++selfNeighborIndex) {
#ifdef STLIB_DEBUG
      assert(neighborMasks.size(i) * MaskDigits == positions.size());
#endif
      const Point& p = _order.position(i);
      const __m128 px = _mm_set1_ps(p[0]);
      const __m128 py = _mm_set1_ps(p[1]);
      const __m128 pz = _mm_set1_ps(p[2]);
      const __m128 intDist =
        _mm_set1_ps(_order.squaredInteractionDistance());
      __m128 x, y, z, d;
      Mask mask;

      const float* block = &shuffled[0];
      // Loop over the masks.
      for (std::size_t j = 0; j != neighborMasks.size(i); ++j) {
        // The lower 4 bits of the mask.
        x = _mm_load_ps(block);
        y = _mm_load_ps(block + 4);
        z = _mm_load_ps(block + 8);
        block += 12;
        d = (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
        // Less than or equal to the interaction distance.
        d = _mm_cmple_ps(d, intDist);
        // Create a 4-bit mask from the most significant bits of the
        // four SP FP values.
        mask = _mm_movemask_ps(d);

        // The upper 4 bits of the mask.
        x = _mm_load_ps(block);
        y = _mm_load_ps(block + 4);
        z = _mm_load_ps(block + 8);
        block += 12;
        d = (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
        // Less than or equal to the interaction distance.
        d = _mm_cmple_ps(d, intDist);
        // Create a 4-bit mask from the most significant bits of the
        // four SP FP values.
        mask |= _mm_movemask_ps(d) << 4;

        neighborMasks(i, j) = mask;
      }

      // Mark the particle itself as not being a neighbor.
#ifdef STLIB_DEBUG
      assert(selfNeighborIndex < neighborMasks.size(i) * MaskDigits);
      assert(p == positions[selfNeighborIndex]);
#endif
      neighborMasks(i, selfNeighborIndex / MaskDigits) &=
        ~(Mask(1) << (selfNeighborIndex % MaskDigits));
#ifdef STLIB_DEBUG
      assert(! isNeighbor(i, selfNeighborIndex));
#endif
    }
  }
}
#endif // __SSE__
#endif // 0


#ifdef _OPENMP
template<typename _Order>
inline
void
AdjacentMask<_Order>::
_partitionCells(const std::size_t localBegin, const std::size_t localEnd,
                std::vector<std::size_t>* delimiters) const
{
  delimiters->clear();
  // The weights are the number of adjacent neighbors for each local cell.
  std::vector<std::size_t>
  weights(_order.numAdjacentNeighbors.begin() + localBegin,
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
AdjacentMask<_Order>::
printMemoryUsageTable(std::ostream& out) const
{
  out << ",used,capacity\n"
      << "neighbors masks,"
      << neighborMasks.memoryUsage() << ','
      << neighborMasks.memoryCapacity() << '\n';
}


template<typename _Order>
inline
void
AdjacentMask<_Order>::
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
AdjacentMask<_Order>::
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

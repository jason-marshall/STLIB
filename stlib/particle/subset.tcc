// -*- C++ -*-

#if !defined(__particle_subset_tcc__)
#error This file is an implementation detail of subset.
#endif

namespace stlib
{
namespace particle
{


template<typename _Order>
inline
SubsetUnionNeighbors<_Order>::
SubsetUnionNeighbors(const _Order& order) :
  Base(order),
  NeighborsPerformance(),
  unionNeighbors(),
  neighborMasks(),
  _adjacentCells(),
  _cellListIndices()
{
}


template<typename _Order>
inline
std::size_t
SubsetUnionNeighbors<_Order>::
numNeighbors() const
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != neighborMasks.size(); ++i) {
    count += numNeighbors(i);
  }
  return count;
}


template<typename _Order>
inline
std::size_t
SubsetUnionNeighbors<_Order>::
numNeighbors(const std::size_t i) const
{
  std::size_t count = 0;
  for (std::size_t j = 0; j != neighborMasks.size(i); ++j) {
    count += numerical::popCount(neighborMasks(i, j));
  }
  return count;
}


template<typename _Order>
template<typename _Allocator1, typename _Allocator2>
inline
void
SubsetUnionNeighbors<_Order>::
potentialNeighbors(const std::size_t particle,
                   std::vector<std::size_t, _Allocator1>* indices,
                   std::vector<Point, _Allocator2>* positions) const
{
  // Resize the vectors.
  indices->resize(neighborMasks.size(particle) * MaskDigits);
  positions->resize(neighborMasks.size(particle) * MaskDigits);

  // Set the indices and positions.
  const std::size_t listIndex = _cellListIndices[particle];
  // For each of the potential neighbors.
  const std::size_t unionSize = unionNeighbors.size(listIndex);
  for (std::size_t i = 0; i != unionSize; ++i) {
    const Neighbor& neighbor = unionNeighbors(listIndex, i);
    (*indices)[n] = neighbor.particle;
    (*positions)[n] = Base::_neighborPosition(neighbor);
  }

  // Pad the indices with zero and the positions with NaN's.
  std::fill(indices->begin() + unionSize, indices->end(), 0);
  std::fill(positions->begin() + unionSize, positions->end(),
            ext::filled_array<Point>(std::numeric_limits<Float>::quiet_NaN()));
}


template<typename _Order>
inline
bool
SubsetUnionNeighbors<_Order>::
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


// Don't use this for counting the neighbors for a single particle as it
// is slower than the simple method.
template<typename _Order>
inline
std::size_t
SubsetUnionNeighbors<_Order>::
popCount(const Mask* masks, std::size_t size) const
{
  // Process a block at a time.
  const std::size_t BlockSize = sizeof(unsigned long) / sizeof(Mask);
  unsigned long block;
  std::size_t count = 0;
  for (; size >= BlockSize; size -= BlockSize, masks += BlockSize) {
    block = 0;
    for (std::size_t i = 0; i != BlockSize; ++i) {
      block <<= std::numeric_limits<Mask>::digits;
      block |= masks[i];
    }
    count += numerical::popCount(block);
  }

  // Finish off the tail.
  block = 0;
  for (std::size_t i = 0; i != size; ++i) {
    block <<= std::numeric_limits<Mask>::digits;
    block |= masks[i];
  }
  count += numerical::popCount(block);

  return count;
}


template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
findPotentialNeighbors(const std::size_t localCellsBegin,
                       const std::size_t localCellsEnd)
{
  _timer.start();
  assert(localCellsBegin <= _order.cellsSize() &&
         localCellsBegin <= localCellsEnd &&
         localCellsEnd <= _order.cellsSize());
  ++_potentialNeighborsCount;
  _cellsBegin = localCellsBegin;
  _cellsEnd = localCellsEnd;

  // Clear the sequence of particles in adjacent cells.
  _adjacentCells.clear();

  const AdjacentList Empty =
    ext::filled_array<AdjacentList>(NeighborRange::Null());

  // CONTINUE: Use cell indices.
  const std::size_t particlesBegin = _order.cellBegin(Base::_cellsBegin);
  const std::size_t particlesEnd = _order.cellBegin(Base::_cellsEnd);

#ifdef _OPENMP
  // Determine the range of particles for each thread.
  std::vector<std::size_t> delimiters;
  _partitionByCells(particlesBegin, particlesEnd, &delimiters);
  // Make a vector of adjacency lists, one for each thread.
  std::vector<std::vector<AdjacentList> >
  adjacentCellsForThread(omp_get_max_threads());
  // Add an empty list for the foreign particles that precede and follow
  // the local ones.
  adjacentCellsForThread[0].pushArray();
  #pragma omp parallel default(none) shared(adjacentCellsForThread, delimiters)
  {
    // For threaded implementations, reference one of the packed arrays.
    std::vector<AdjacentList>* adjacent =
      &adjacentCellsForThread[omp_get_thread_num()];
    const std::size_t begin = delimiters[omp_get_thread_num()];
    const std::size_t end = delimiters[omp_get_thread_num() + 1];

    AdjacentList adj;
    // Loop over the cells.
    for (std::size_t i = begin; i < end;) {
      // Determine the adjacent cells.
      Base::_adjacentCells(i, &adj);
      // Record them.
      adjacent->push_back(adj);
      // Advance to the next cell.
      const Code code = _order.codes()[i];
      for (; _order.codes()[i] == code && i < end; ++i) {
      }
    }
  }
  // Merge the adjacent cells from the threads.
  for (std::size_t i = 0; i != adjacentCellsForThread.size(); ++i) {
    _adjacentCells.insert(_adjacentCells.end(),
                          adjacentCellsForThread[i].begin(),
                          adjacentCellsForThread[i].end());
  }
#else
  _adjacentCells.pushArray();
  AdjacentList adj;
  // Loop over the cells.
  for (std::size_t i = particlesBegin; i < particlesEnd;) {
    // Determine the adjacent cells.
    Base::_adjacentCells(i, &adj);
    // Record them.
    _adjacentCells.push_back(adj);
    // Advance to the next cell.
    const Code code = _order.codes()[i];
    for (; _order.codes()[i] == code && i < particlesEnd; ++i) {
    }
  }
#endif

  //
  // Set values for _cellListIndices.
  //
  _cellListIndices.resize(_order.particles.size());
  // Record the list indices for the particles that precede the local ones.
  std::fill(_cellListIndices.begin(),
            _cellListIndices.begin() + particlesBegin, 0);
  std::size_t n = 1;
  // For each cell.
  for (std::size_t i = particlesBegin; i != particlesEnd; ++n) {
    const Code code = _order.codes()[i];
    // For each particle in the cell.
    for (; _order.codes()[i] == code && i != particlesEnd; ++i) {
      _cellListIndices[i] = n;
    }
  }
  assert(n == _adjacentCells.numArrays());
  // Record the list indices for the particles that follow the local ones.
  std::fill(_cellListIndices.begin() + particlesEnd,
            _cellListIndices.end(), 0);

  _timer.stop();
  _timePotentialNeighbors += _timer.elapsed();
}


// CONTINUE: Use a bounding ball that contains the particles to exclude
// potential neighbors. (Note that the particles assigned to a cell may not
// actually lie within the cell.) Let c_i be the distance from particle i to the
// center of the ball and b be the radius of the ball. We can get a lower
// bound on the distance from potential neighbor i to any of the particles
// in a cell.
// d_i >= c_i - b
// We can exclude i as being a neighbor if
// c_i - b > r
// where r is the interaction distance. We can write this in terms of
// the squared distance from the center of the ball to particle i.
// c_i^2 > (r + b)^2
//
// Any particle in the center cell is automatically included as a potential
// neighbor. If a particle is not in the center cell or is not excluded using
// the lower bound, we need to calculate distances to the particles in the
// cell. We iterate until we find a distance that is less than or equal to
// the cutoff. If such a distance is found, the point is in the union,
// otherwise not. We pack the positions of the center particles into aligned
// memory and shuffle the coordinates so that we can use vertical SIMD
// operations. For the sake of efficiency, we need to change the ordering
// of the particles. With sub-cell ordering, their postions are correlated.
// Simply dividing the center particles into four parts and then interleaving
// will do the trick.
template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
findNeighbors()
{
  _timer.start();
  ++_neighborsCount;
  assert(_cellListIndices.size() == _order.particles.size());

  // Cache the particle positions.
  _order.recordPositions();

  // CONTINUE: Use cell indices.
  const std::size_t particlesBegin = _order.cellBegin(Base::_cellsBegin);
  const std::size_t particlesEnd = _order.cellBegin(Base::_cellsEnd);

#ifdef _OPENMP
  // Determine the range of particles for each thread.
  std::vector<std::size_t> delimiters;
  _partitionByCells(particlesBegin, particlesEnd, &delimiters);
  #pragma omp parallel default(none) shared(delimiters)
  {
    _findNeighbors(delimiters[omp_get_thread_num()],
                   delimiters[omp_get_thread_num() + 1]);
  }
#else
  _findNeighbors(particlesBegin, particlesEnd);
#endif
  _timer.stop();
  _timeNeighbors += _timer.elapsed();
}


// CONTINUE HERE
template<typename _Order>
template<typename _Float, typename _Dimension>
inline
void
SubsetUnionNeighbors<_Order>::
_findNeighbors(const std::size_t begin, const std::size_t end,
               _Float /*dummy*/, _Dimension /*dummy*/)
{
  // CONTINUE HERE: Move to findNeighbors().
#if 0
  //
  // Set zero bit-masks for neighborMasks.
  //
  // The number of masks for each particles.
  std::vector<std::size_t> sizes(_order.particles.size());
  // The particles that precede the local ones.
  std::fill(sizes.begin(), sizes.begin() + localBegin, 0);
  // For each cell.
  for (std::size_t i = localBegin; i != localEnd;) {
    // The required number of masks for the given number of potential
    // neighbors.
    const std::size_t numPot = _countPotentialNeighbors(i);
    const std::size_t numMasks = (numPot + MaskDigits - 1) / MaskDigits;
    const Code code = _order.codes()[i];
    // For each particle in the cell.
    for (; _order.codes()[i] == code && i != localEnd; ++i) {
      // The number of masks needed for the particle.
      sizes[i] = numMasks;
    }
  }
  std::fill(sizes.begin() + localEnd, sizes.end(), 0);
  // Make the packed array from the sizes.
  neighborMasks.rebuild(sizes.begin(), sizes.end());
  // Fill it with zero values.
  std::fill(neighborMasks.begin(), neighborMasks.end(), Mask(0));
#endif


  // The index of the particle itself in the neighbor list.
  std::size_t selfNeighborIndex;
  // The positions of the potential neighbors for a cell.
  std::vector<Point> positions;
  // Loop over the cells.
  for (std::size_t i = begin; i != end; /*No increment.*/) {
    // Determine the neighbor index of the particle itself so it can be
    // marked as not being a neighbor.
    selfNeighborIndex = _selfNeighborIndex(i);
    // Get the positions of the potential neighbors.
    _extractPotentialNeighborPositions(i, &positions);
    // Loop over the particles in the cell.
    const Code code = _order.codes()[i];
    for (; _order.codes()[i] == code; ++i, ++selfNeighborIndex) {
#ifdef STLIB_DEBUG
      assert(neighborMasks.size(i) * MaskDigits == positions.size());
#endif
      std::size_t index = 0;
      // Loop over the masks.
      for (std::size_t j = 0; j != neighborMasks.size(i); ++j) {
        Mask mask = 0;
        Mask bit = 1;
        for (std::size_t k = 0; k != MaskDigits;
             ++k, ++index, bit <<= 1) {
          if (_isNeighbor(i, positions, index)) {
            mask |= bit;
          }
        }
        neighborMasks(i, j) = mask;
      }

      // Mark the particle itself as not being a neighbor.
#ifdef STLIB_DEBUG
      assert(selfNeighborIndex < neighborMasks.size(i) * MaskDigits);
      assert(_order.position(i) == positions[selfNeighborIndex]);
#endif
      neighborMasks(i, selfNeighborIndex / MaskDigits) &=
        ~(Mask(1) << (selfNeighborIndex % MaskDigits));
#ifdef STLIB_DEBUG
      assert(! isNeighbor(i, selfNeighborIndex));
#endif
    }
  }
}


// CONTINUE: Implement.
#if 0
#ifdef __SSE__
template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
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
      assert(_order.position(i) == positions[selfNeighborIndex]);
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
SubsetUnionNeighbors<_Order>::
_partitionByCells(const std::size_t localBegin, const std::size_t localEnd,
                  std::vector<std::size_t>* delimiters) const
{
  delimiters->clear();
  // Determine the range of particles for each thread.
  // Start with a fair partition.
  numerical::computePartitions(localEnd - localBegin,
                               std::size_t(omp_get_max_threads()),
                               std::back_inserter(*delimiters));
  for (std::size_t i = 0; i != delimiters->size(); ++i) {
    (*delimiters)[i] += localBegin;
  }
  // Adjust the delimiters to respect cell boundaries.
  for (std::size_t i = delimiters->size() - 2; i > 0 ; --i) {
    while ((*delimiters)[i] != (*delimiters)[i + 1] &&
           _order.codes()[(*delimiters)[i]] ==
           _order.codes()[(*delimiters)[i] - 1]) {
      ++(*delimiters)[i];
    }
  }
}
#endif


template<typename _Order>
inline
std::size_t
SubsetUnionNeighbors<_Order>::
_selfNeighborIndex(const std::size_t i) const
{
  std::size_t selfNeighborIndex = 0;
  const std::size_t listIndex = _cellListIndices[i];
  // For each contiguous range.
  for (std::size_t j = 0; j != _adjacent.size(listIndex); ++j) {
    const NeighborRange& range = _adjacent(listIndex, j);
    // If the particle is in this range.
    if (range.first.particle + range.extent > i) {
      selfNeighborIndex += i - range.first.particle;
      break;
    }
    else {
      selfNeighborIndex += range.extent;
    }
  }
  return selfNeighborIndex;
}


template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
_extractPotentialNeighborPositions(std::size_t particle,
                                   std::vector<Point>* positions) const
{
  // Use the bit-masks to determine the (padded) number of potential neighbors.
  positions->resize(neighborMasks.size(particle) * MaskDigits);
  // Extract the positions.
  const std::size_t listIndex = _cellListIndices[particle];
  typename std::vector<Point>::pointer p = &(*positions)[0];
  // For each contiguous range.
  for (std::size_t i = 0; i != _adjacent.size(listIndex); ++i) {
    // Copy the range of positions.
    const NeighborRange& range = _adjacent(listIndex, i);
    memcpy(p, &_order.positions[range.first.particle],
           range.extent * sizeof(Point));
    // For periodic domains, we may need to offset the positions.
    Base::_offsetPositions(p, range);
    p += range.extent;
  }
#ifdef STLIB_DEBUG
  assert(p <= &positions->back() + 1);
#endif
  // Pad the positions with NaN points.
  std::fill(p, &positions->back() + 1,
            ext::filled_array<Point>(std::numeric_limits<Float>::quiet_NaN()));
}


template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
printMemoryUsageTable(std::ostream& out) const
{
  out << ",used,capacity\n"
      << "adjacent,"
      << _adjacent.memoryUsage() << ','
      << _adjacent.memoryCapacity() << '\n'
      << "adjacentListIndices,"
      << _cellListIndices.size() * sizeof(std::size_t) << ','
      << _cellListIndices.capacity() * sizeof(std::size_t) << '\n'
      << "neighbors masks,"
      << neighborMasks.memoryUsage() << ','
      << neighborMasks.memoryCapacity() << '\n';
}


template<typename _Order>
inline
void
SubsetUnionNeighbors<_Order>::
printInfo(std::ostream& out) const
{
  std::size_t numWithNeighbors = 0;
  for (std::size_t i = 0; i != _order.particles.size(); ++i) {
    numWithNeighbors += ! _adjacent.empty(_cellListIndices[i]);
  }
  out << "Dimension = " << Dimension << '\n'
      << "Number of particles = " << _order.particles.size() << '\n'
      << "Number with neighbors = " << numWithNeighbors << '\n';
}


} // namespace particle
}

// -*- C++ -*-

#if !defined(__particle_verlet_tcc__)
#error This file is an implementation detail of verlet.
#endif

namespace stlib
{
namespace particle
{


template<typename _Order>
inline
VerletLists<_Order>::
VerletLists(const _Order& order) :
  NeighborsPerformance(),
  _order(order),
  neighbors()
{
}


template<typename _Order>
inline
void
VerletLists<_Order>::
getNeighbors(const std::size_t particle, std::vector<std::size_t>* indices,
             std::vector<Point>* positions) const
{
  indices->resize(neighbors.size(particle));
  positions->resize(neighbors.size(particle));
  for (std::size_t i = 0; i != neighbors.size(particle); ++i) {
    const Neighbor& neighbor = neighbors(particle, i);
    indices[i] = neighbor.particle;
    positions[i] = _order.neighborPosition(neighbor);
  }
}


template<typename _Order>
inline
void
VerletLists<_Order>::
_findNeighbors(const std::size_t localCellsBegin,
               const std::size_t localCellsEnd)
{
  _timer.start();
  assert(localCellsBegin <= _order.cellsSize() &&
         localCellsBegin <= localCellsEnd &&
         localCellsEnd <= _order.cellsSize());
  ++_neighborsCount;

  // Clear the sequence of neighbors.
  neighbors.clear();

  // The delimiters for the particles.
  const std::size_t particlesBegin = _order.cellBegin(localCellsBegin);
  const std::size_t particlesEnd = _order.cellBegin(localCellsEnd);

#ifdef _OPENMP
  // Make a vector of packed arrays, one for each thread.
  std::vector<container::PackedArrayOfArrays<Neighbor> >
  neighborsForThread(omp_get_max_threads());
  // Add empty arrays for the shadow particles that precede the local ones.
  neighborsForThread[0].pushArrays(particlesBegin);
  #pragma omp parallel default(none) shared(neighborsForThread)
  {
    // Reference one of the packed arrays.
    container::PackedArrayOfArrays<Neighbor>* neighborsRef =
      &neighborsForThread[omp_get_thread_num()];
    // Determine the range of cells for this thread.
    std::size_t begin, end;
    numerical::getPartitionRange(localCellsEnd - localCellsBegin,
                                 &begin, &end);
    begin += localCellsBegin;
    end += localCellsBegin;
    _findNeighbors(begin, end, neighborsRef, Float(0),
                   std::integral_constant<std::size_t, Dimension>());
  }
  // Merge the neighbors from the threads.
  neighbors.rebuild(neighborsForThread);
#else
  neighbors.pushArrays(particlesBegin);
  _findNeighbors(localCellsBegin, localCellsEnd, &neighbors, Float(0),
                 std::integral_constant<std::size_t, Dimension>());
#endif

  // Add empty arrays for the shadow particles that follow the local ones.
  neighbors.pushArrays(_order.particles.size() - particlesEnd);

#ifdef STLIB_DEBUG
  assert(neighbors.numArrays() == _order.particles.size());
#endif
  _timer.stop();
  _timeNeighbors += _timer.elapsed();
}


template<typename _Order>
template<typename _Float, typename _Dimension>
inline
void
VerletLists<_Order>::
_findNeighbors(const std::size_t cellsBegin, const std::size_t cellsEnd,
               container::PackedArrayOfArrays<Neighbor>* neighborsRef,
               _Float /*dummy*/, _Dimension /*dummy*/)
{
  // Work with the squared distance.
  const Float squaredRadius = _order.squaredInteractionDistance();

  // Indices and positions for the particles in the adjacent cells.
  std::vector<Neighbor> indices;
  std::vector<Point> adjPos;
  // Loop over the cells.
  for (std::size_t cell = cellsBegin; cell != cellsEnd; ++cell) {
    // Determine the indices and positions for the particles in the
    // adjacent cells.
    _order.positionsInAdjacent(cell, &indices, &adjPos);
    // Loop over the particles in this cell.
    for (std::size_t particle = _order.cellBegin(cell);
         particle != _order.cellEnd(cell); ++particle) {
      const Point pos = _order.position(particle);
      // Start a new array of neighbors for this particle.
      neighborsRef->pushArray();
      // Loop over the ajacent particles.
      for (std::size_t i = 0; i != indices.size(); ++i) {
        // Exclude the particle itself. Check the distance.
        if (indices[i].particle != particle &&
            ext::squaredDistance(pos, adjPos[i]) <= squaredRadius) {
          neighborsRef->push_back(indices[i]);
        }
      }
    }
  }
}


// If either AVX or SSE is supported.
#ifdef __SSE__
template<typename _Order>
inline
void
VerletLists<_Order>::
_findNeighbors(const std::size_t cellsBegin, const std::size_t cellsEnd,
               container::PackedArrayOfArrays<Neighbor>* packedNeighbors,
               float /*dummy*/,
               std::integral_constant<std::size_t, 3> /*3D*/)
{
  typedef simd::Vector<float>::Type Vector;
  // The number of single-precision floats in the register.
  const std::size_t Block = simd::Vector<float>::Size;
  // Work with the squared distance.
  const Vector intDist =
    simd::set1(_order.squaredInteractionDistance());
  const Point nanPoint = ext::filled_array<Point>
                         (std::numeric_limits<Float>::quiet_NaN());
  // Indices and positions for the particles in the adjacent cells.
  std::vector<Neighbor> neighbors;
  std::vector<Point> positions;
  // The shuffled coordinates.
  std::vector<float, simd::allocator<float> > shuffled;
  Vector x, y, z, d;
  unsigned mask;
  // Loop over the cells.
  for (std::size_t cell = cellsBegin; cell != cellsEnd; ++cell) {
    const std::size_t offset = _order.centerCellOffset(cell);
    // Determine the indices and positions for the particles in the
    // adjacent cells.
    _order.positionsInAdjacent(cell, &neighbors, &positions);
    // Pad the positions with NaN.
    positions.insert(positions.end(), (positions.size() + Block - 1) /
                     Block * Block - positions.size(),
                     nanPoint);
    const std::size_t numBlocks = positions.size() / Block;
    // Shuffle the coordinates so that we can use SSE operations.
    simd::aosToHybridSoa(positions, &shuffled);
    // Loop over the particles in this cell.
    for (std::size_t particle = _order.cellBegin(cell);
         particle != _order.cellEnd(cell); ++particle) {
      // Record the coordinates for the particle.
      const Point p = _order.position(particle);
      const Vector px = simd::set1(p[0]);
      const Vector py = simd::set1(p[1]);
      const Vector pz = simd::set1(p[2]);
      // Invalidate the particle's position so that it will not be recorded
      // as its own neighbor.
      const std::size_t self = particle - _order.cellBegin(cell) + offset;
      const std::size_t n = self / Block;
      const std::size_t m = self % Block;
      shuffled[Dimension * Block * n + m] =
        std::numeric_limits<Float>::quiet_NaN();
      shuffled[Dimension * Block * n + m + Block] =
        std::numeric_limits<Float>::quiet_NaN();
      shuffled[Dimension * Block * n + m + 2 * Block] =
        std::numeric_limits<Float>::quiet_NaN();
      // Start a new array of neighbors for this particle.
      packedNeighbors->pushArray();
      const float* block = &shuffled[0];
      for (std::size_t i = 0; i != numBlocks; ++i) {
        x = simd::load(block);
        y = simd::load(block + Block);
        z = simd::load(block + 2 * Block);
        block += Dimension * Block;
        d = (px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z);
        // Less than or equal to the squared interaction distance.
        d = simd::lessEqual(d, intDist);
        // Create a mask from the most significant bits of the
        // SP FP values.
        mask = simd::moveMask(d);
        // Check the distance to determine neighbors.
        for (std::size_t j = 0; j != Block; ++j, mask >>= 1) {
          if (mask & 1) {
            packedNeighbors->push_back(neighbors[Block * i + j]);
          }
        }
      }
      // Restore the particle's position.
      shuffled[Dimension * Block * n + m] = positions[self][0];
      shuffled[Dimension * Block * n + m + Block] = positions[self][1];
      shuffled[Dimension * Block * n + m + 2 * Block] = positions[self][2];
    }
  }
}
#endif

template<typename _Order>
inline
void
VerletLists<_Order>::
printInfo(std::ostream& out) const
{
  std::size_t numWithNeighbors = 0;
  for (std::size_t i = 0; i != neighbors.numArrays(); ++i) {
    numWithNeighbors += ! neighbors.empty(i);
  }
  out << "Dimension = " << Dimension << '\n'
      << "Number of particles = " << _order.particles.size() << '\n'
      << "Number with neighbors = " << numWithNeighbors << '\n'
      << "Total # of neighbors = " << neighbors.size() << '\n';
}


template<typename _Order>
inline
void
VerletLists<_Order>::
_printMemoryUsageTable(std::ostream& out) const
{
  out << ",used,capacity\n"
      << "neighbors,"
      << neighbors.memoryUsage() << ','
      << neighbors.memoryCapacity() << '\n';
}


} // namespace particle
}

// -*- C++ -*-

#if !defined(__particle_verletPotential_tcc__)
#error This file is an implementation detail of verlet.
#endif

namespace stlib
{
namespace particle
{


template<typename _Order>
inline
VerletListsPotential<_Order>::
VerletListsPotential(const _Order& order) :
  NeighborsPerformance(),
  _order(order),
  _cellsBegin(-1),
  _cellsEnd(-1),
  _potentialNeighbors(),
  _neighbors()
{
}


template<typename _Order>
inline
bool
VerletListsPotential<_Order>::
_isNeighbor(const std::size_t particle, const std::size_t index) const
{
#ifdef STLIB_DEBUG
  assert(particle < _order.particles.size());
  assert(index < numPotentialNeighbors(particle));
#endif
  return ext::squaredDistance(_order.position(particle),
                              potentialNeighborPosition(particle, index)) <=
    _order.squaredInteractionDistance();
}


template<typename _Order>
inline
void
VerletListsPotential<_Order>::
_findPotentialNeighbors(const std::size_t localCellsBegin,
                        const std::size_t localCellsEnd)
{
  // Work with the squared distance.
  const Float squaredRadius =
    (_order.interactionDistance() + _order.padding()) *
    (_order.interactionDistance() + _order.padding());

  // Clear the sequence of potential neighbors.
  _potentialNeighbors.clear();

  // The delimiters for the particles.
  const std::size_t localBegin = _order.cellBegin(localCellsBegin);
  const std::size_t localEnd = _order.cellBegin(localCellsEnd);
#ifdef _OPENMP
  // Make a vector of packed arrays, one for each thread.
  std::vector<container::PackedArrayOfArrays<Neighbor> >
  neighborsForThread(omp_get_max_threads());
  // Add empty arrays for the foreign particles that precede the local ones.
  neighborsForThread[0].pushArrays(localBegin);
#else
  _potentialNeighbors.pushArrays(localBegin);
#endif


#ifdef _OPENMP
  #pragma omp parallel default(none) shared(neighborsForThread)
  {
    // For threaded implementations, reference one of the packed arrays.
    container::PackedArrayOfArrays<Neighbor>* neighbors =
      &neighborsForThread[omp_get_thread_num()];
#else
  // For serial code, reference _potentialNeighbors.
  container::PackedArrayOfArrays<Neighbor>* neighbors =
    &_potentialNeighbors;
#endif

    // Determine the range of cells for this thread.
    std::size_t begin, end;
    numerical::getPartitionRange(localCellsEnd - localCellsBegin,
                                 &begin, &end);
    begin += localCellsBegin;
    end += localCellsBegin;
    // Indices and positions for the particles in the adjacent cells.
    std::vector<Neighbor> indices;
    std::vector<Point> adjPos;
    // Loop over the cells.
    for (std::size_t cell = begin; cell != end; ++cell) {
      // Determine the indices and positions for the particles in the
      // adjacent cells.
      _order.positionsInAdjacent(cell, &indices, &adjPos);
      // Loop over the particles in this cell.
      for (std::size_t particle = _order.cellBegin(cell);
           particle != _order.cellEnd(cell); ++particle) {
        const Point pos = _order.position(particle);
        // Start a new array of potential neighbors for this particle.
        neighbors->pushArray();
        // Loop over the ajacent particles.
        for (std::size_t i = 0; i != indices.size(); ++i) {
          // Exclude the particle itself. Check the distance.
          if (indices[i].particle != particle &&
              ext::squaredDistance(pos, adjPos[i]) <= squaredRadius) {
            neighbors->push_back(indices[i]);
          }
        }
      }
    }
#ifdef _OPENMP
  }
  // Merge the neighbors from the threads.
  _potentialNeighbors.rebuild(neighborsForThread);
#endif

  // Add empty arrays for the foreign particles that follow the local ones.
  _potentialNeighbors.pushArrays(_order.particles.size() - localEnd);

#ifdef STLIB_DEBUG
  assert(_potentialNeighbors.numArrays() == _order.particles.size());
#endif
}


template<typename _Order>
inline
void
VerletListsPotential<_Order>::
findNeighbors()
{
  // The potential neighbors must have been calculated.
  assert(_potentialNeighbors.numArrays() == _order.particles.size());
  assert(_cellsBegin <= _cellsEnd && _cellsEnd <= _order.cellsSize());
  _timer.start();
  ++_neighborsCount;

  // Clear the sequence of neighbors.
  _neighbors.clear();

  // Add empty arrays for the foreign particles that precede the local ones.
  const std::size_t particlesBegin = _order.cellBegin(_cellsBegin);
  const std::size_t particlesEnd = _order.cellBegin(_cellsEnd);

#ifdef _OPENMP
  // Make a vector of packed arrays, one for each thread.
  std::vector<container::PackedArrayOfArrays<Neighbor> >
  neighborsForThread(omp_get_max_threads());
  neighborsForThread[0].pushArrays(particlesBegin);
  #pragma omp parallel default(none) shared(neighborsForThread)
  {
    // Reference one of the packed arrays.
    container::PackedArrayOfArrays<Neighbor>* neighborsRef =
      &neighborsForThread[omp_get_thread_num()];

    // Determine the range of particles for this thread.
    std::size_t begin, end;
    numerical::getPartitionRange(particlesEnd - particlesBegin,
                                 &begin, &end);
    begin += particlesBegin;
    end += particlesBegin;
    // Loop over the local particles.
    for (std::size_t i = begin; i != end; ++i) {
      // Start a new array of neighbors for this particle.
      neighborsRef->pushArray();
      for (std::size_t j = 0; j != numPotentialNeighbors(i); ++j) {
        if (_isNeighbor(i, j)) {
          neighborsRef->push_back(_potentialNeighbors(i, j));
        }
      }
    }
  }
  // Merge the neighbors from the threads.
  _neighbors.rebuild(neighborsForThread);
#else
  _neighbors.pushArrays(particlesBegin);
  // Loop over the local particles.
  for (std::size_t i = particlesBegin; i != particlesEnd; ++i) {
    // Start a new array of neighbors for this particle.
    _neighbors.pushArray();
    for (std::size_t j = 0; j != numPotentialNeighbors(i); ++j) {
      if (_isNeighbor(i, j)) {
        _neighbors.push_back(_potentialNeighbors(i, j));
      }
    }
  }
#endif

  // Add empty arrays for the foreign particles that follow the local ones.
  _neighbors.pushArrays(_order.particles.size() - particlesEnd);
  assert(_neighbors.numArrays() == _potentialNeighbors.numArrays());
  _timer.stop();
  _timeNeighbors += _timer.elapsed();
}


template<typename _Order>
inline
void
VerletListsPotential<_Order>::
printMemoryUsageTable(std::ostream& out) const
{
  out << ",used,capacity\n"
      << "potentialNeighbors,"
      << _potentialNeighbors.memoryUsage() << ','
      << _potentialNeighbors.memoryCapacity() << '\n'
      << "neighbors,"
      << _neighbors.memoryUsage() << ','
      << _neighbors.memoryCapacity() << '\n';
}


template<typename _Order>
inline
void
VerletListsPotential<_Order>::
printInfo(std::ostream& out) const
{
  std::size_t numWithNeighbors = 0;
  for (std::size_t i = 0; i != _potentialNeighbors.numArrays(); ++i) {
    numWithNeighbors += ! _potentialNeighbors.empty(i);
  }
  out << "Dimension = " << Dimension << '\n'
      << "Number of particles = " << _order.particles.size() << '\n'
      << "Number with neighbors = " << numWithNeighbors << '\n'
      << "Total # of potential neighbors = "
      << _potentialNeighbors.size() << '\n'
      << "Total # of neighbors = " << _neighbors.size() << '\n';
}


} // namespace particle
}

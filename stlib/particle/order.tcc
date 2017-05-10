// -*- C++ -*-

#if !defined(__particle_order_tcc__)
#error This file is an implementation detail of order.
#endif

namespace stlib
{
namespace particle
{


template<typename _Traits>
inline
MortonOrder<_Traits>::
MortonOrder(const geom::BBox<Float, Dimension>& domain,
            const Float interactionDistance, const Float padding) :
  particles(),
  adjacentCells(),
  numAdjacentNeighbors(),
  morton(domain, interactionDistance +
         _appropriatePadding(interactionDistance, padding)),
  // Use 2 levels of subcell ordering, if there are enough availabe digits.
  _subcellLevels(std::min(std::size_t(2),
                          (std::numeric_limits<Code>::digits - Dimension*
                           morton.numLevels() - 1) / Dimension)),
  _subcellMorton(morton),
  _cellDelimiters(1),
  _cellCodes(1),
  _lookupTable(),
  _getPosition(),
  _setPosition(),
  _periodicOffsets(),
  _interactionDistance(interactionDistance),
  _squaredInteractionDistance(interactionDistance* interactionDistance),
  _padding(_appropriatePadding(interactionDistance, padding)),
  _startingPositions(),
  _reorderCount(0),
  _repairCount(0),
  _timer(),
  _timeIsOrderValid(0),
  _timeOrder(0),
  _timeRecordStartingPositions(0),
  _timeBuildLookupTable(0),
  _timeAdjacentCells(0)
{
  _subcellMorton.setLevels(morton.numLevels() + _subcellLevels);
  // At least three cells in each direction are required for periodic domains.
  // Otherwise, neighbors would be listed multiple times.
  if (Periodic) {
    assert(ext::min(morton.cellExtents()) >= 3);
  }
  _calculatePeriodicOffsets();
  // Initialize the cell delimiters for an empty set of cells.
  _cellDelimiters[0] = 0;
  // Set the guard element for the cell codes.
  _cellCodes[0] = morton.maxCode() + 1;
}


template<typename _Traits>
inline
MortonOrder<_Traits>::
MortonOrder() :
  particles(),
  adjacentCells(),
  numAdjacentNeighbors(),
  morton(),
  _subcellLevels(0),
  _subcellMorton(),
  _cellDelimiters(1),
  _cellCodes(1),
  _lookupTable(),
  _getPosition(),
  _setPosition(),
  _periodicOffsets(),
  _interactionDistance(std::numeric_limits<Float>::quiet_NaN()),
  _squaredInteractionDistance(std::numeric_limits<Float>::quiet_NaN()),
  _padding(std::numeric_limits<Float>::quiet_NaN()),
  _startingPositions(),
  _reorderCount(0),
  _repairCount(0),
  _timer(),
  _timeIsOrderValid(0),
  _timeOrder(0),
  _timeRecordStartingPositions(0),
  _timeBuildLookupTable(0),
  _timeAdjacentCells(0)
{
  // Initialize the cell delimiters for an empty set of cells.
  _cellDelimiters[0] = 0;
  // Set the guard element for the cell codes.
  _cellCodes[0] = morton.maxCode() + 1;
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
initialize(const geom::BBox<Float, Dimension>& domain,
           const Float interactionDistance, const Float padding)
{
  particles.clear();
  morton.initialize(domain, interactionDistance +
                    _appropriatePadding(interactionDistance, padding));
  // At least three cells in each direction are required for periodic domains.
  // Otherwise, neighbors would be listed multiple times.
  if (Periodic) {
    assert(min(morton.cellExtents()) >= 3);
  }
  _calculatePeriodicOffsets();
  _subcellMorton = morton;
  // Use 2 levels of subcell ordering, if there are enough availabe digits.
  _subcellLevels = std::min(std::size_t(2),
                            (std::numeric_limits<Code>::digits - Dimension *
                             morton.numLevels() - 1) / Dimension);
  _subcellMorton.setLevels(morton.numLevels() + _subcellLevels);
  // Initialize the cell delimiters for an empty set of cells.
  _cellDelimiters.resize(1);
  _cellDelimiters[0] = 0;
  // Set the guard element for the cell codes.
  _cellCodes[0] = morton.maxCode() + 1;
  adjacentCells.clear();
  numAdjacentNeighbors.clear();
  // Set parameters.
  _interactionDistance = interactionDistance;
  _squaredInteractionDistance = interactionDistance * interactionDistance;
  _padding = _appropriatePadding(interactionDistance, padding);
  _startingPositions.clear();
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
_calculatePeriodicOffsets()
{
  typedef std::array<std::size_t, Dimension> DiscCoord;
  typedef container::SimpleMultiIndexRange<Dimension> IndexRange;
  typedef container::SimpleMultiIndexRangeIterator<Dimension> IndexIterator;

  const IndexRange range = {ext::filled_array<DiscCoord>(3),
                            ext::filled_array<DiscCoord>(0)
                           };
  const IndexIterator end = IndexIterator::end(range);
  std::size_t n = 0;
  for (IndexIterator i = IndexIterator::begin(range); i != end; ++i, ++n) {
    _periodicOffsets[n] = (ext::convert_array<Float>(*i) - Float(1)) *
                          lengths();
  }
  assert(n == _periodicOffsets.size());
}


template<typename _Traits>
inline
typename MortonOrder<_Traits>::Float
MortonOrder<_Traits>::
_appropriatePadding(Float interactionDistance, Float padding)
{
  if (padding != padding) {
    // With this value of the padding one would expect twice as many
    // potential neighbors as actual neighbors.
    padding = (std::pow(2, 1. / 3) - 1) * interactionDistance;
  }
  return padding;
}


template<typename _Traits>
template<typename _InputIterator>
inline
void
MortonOrder<_Traits>::
setParticles(_InputIterator begin, _InputIterator end)
{
  // Record the unordered particles.
  particles.clear();
  particles.insert(particles.end(), begin, end);
  // Order the particles.
  reorder();
}


template<typename _Traits>
inline
bool
MortonOrder<_Traits>::
repair()
{
  ++_repairCount;
  if (! isOrderValid()) {
    reorder();
    return true;
  }
  return false;
}


template<typename _Traits>
inline
bool
MortonOrder<_Traits>::
isOrderValid(const std::size_t begin, const std::size_t end) const
{
  // If the padding is zero, we need to reorder at each time step.
  if (_padding == 0) {
    assert(_startingPositions.empty());
    return false;
  }

  _timer.start();
  assert(particles.size() == _startingPositions.size());
  assert(begin <= particles.size());
  assert(begin <= end && end <= particles.size());
#ifdef STLIB_DEBUG
  // Check that the starting positions are not NaN's.
  for (std::size_t i = begin; i != end; ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      assert(_startingPositions[i][j] == _startingPositions[i][j]);
    }
  }
#endif
  // The maximum allowed distance for a particle to move is half of the
  // padding. If all particles have moved less than half the padding, then the
  // distance between any pair of particles has changed less than the padding.
  const Float maxSquaredDistance = 0.25 * _padding * _padding;
  bool result = true;
  #pragma omp parallel for default(none) reduction(&& : result)
  for (std::ptrdiff_t i = begin; i < std::ptrdiff_t(end); ++i) {
    result = result && (ext::squaredDistance(_getPosition(particles[i]),
                                             _startingPositions[i]) <=
                        maxSquaredDistance);
  }
  _timer.stop();
  _timeIsOrderValid += _timer.elapsed();
  return result;
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
reorder()
{
  ++_reorderCount;
  // Calculate codes from the positions and order the particles.
  order();
  // Record the starting positions for the ordered particles.
  recordStartingPositions();
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
order()
{
  _timer.start();
  // Normalize the positions. (This only has an effect for periodic domains.)
  normalizePositions();

  // Make a vector of code/index pairs.
  std::vector<std::pair<Code, std::size_t> > pairs(particles.size());
  #pragma omp parallel for default(none) shared(pairs)
  for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(pairs.size()); ++i) {
    // Use subcell ordering.
    pairs[i].first = _subcellMorton.code(_getPosition(particles[i]));
    pairs[i].second = i;
  }

  // Technically, it sorts using the composite number formed by the pair.
  // Since the first is most significant, this is fine.
  std::sort(pairs.begin(), pairs.end());

  // Shift to convert to regular Morton codes.
  const int shift = _subcellLevels * Dimension;
  #pragma omp parallel for default(none) shared(pairs)
  for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(pairs.size()); ++i) {
    pairs[i].first >>= shift;
  }

  // Set the order of the particles.
  {
    std::vector<Particle> tmp(particles.size());
    #pragma omp parallel for default(none) shared(pairs, tmp)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(tmp.size()); ++i) {
      tmp[i] = particles[pairs[i].second];
    }
    particles.swap(tmp);
  }

  // Invalidate the lookup table.
  _lookupTable.clear();

  // Set the cell delimiters from the sorted codes.
  _cellDelimiters.clear();
  _cellCodes.clear();
  for (std::size_t i = 0; i != pairs.size(); /*no increment*/) {
    _cellDelimiters.push_back(i);
    const Code code = pairs[i].first;
    _cellCodes.push_back(code);
    // Advance to the next cell.
    while (i != pairs.size() && pairs[i].first == code) {
      ++i;
    }
  }
  _cellDelimiters.push_back(pairs.size());
  _cellCodes.push_back(morton.maxCode() + 1);

  _timer.stop();
  _timeOrder += _timer.elapsed();

  // Build the lookup table for accessing cells.
  buildLookupTable();
  // Build the vector of adjacent cells.
  calculateAdjacentCells();
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
recordStartingPositions()
{
  // Do nothing if the padding is zero.
  if (_padding == 0) {
    return;
  }

  _timer.start();
  _startingPositions.resize(particles.size());
  for (std::size_t i = 0; i != _startingPositions.size(); ++i) {
    _startingPositions[i] = _getPosition(particles[i]);
  }
  _timer.stop();
  _timeRecordStartingPositions += _timer.elapsed();
}


template<typename _Traits>
inline
std::pair<std::size_t, std::size_t>
MortonOrder<_Traits>::
insertCells(const std::vector<std::pair<Code, std::size_t> >& codesSizes,
            const Code localCodesBegin, const Code localCodesEnd)
{
  // Deal with the trivial case.
  if (codesSizes.empty()) {
    return std::make_pair(std::size_t(0), cellsSize());
  }
  // Make a list of the cells that we currently have.
  std::vector<std::pair<Code, std::size_t> > cells(cellsSize());
  for (std::size_t i = 0; i != cells.size(); ++i) {
    cells[i] = std::make_pair(_cellCodes[i],
                              _cellDelimiters[i + 1] - _cellDelimiters[i]);
  }
  // Add the cells to be inserted.
  cells.insert(cells.end(), codesSizes.begin(), codesSizes.end());
  // Sort by the cell codes.
  std::sort(cells.begin(), cells.end());

  // Determine the range of original (local) cells in the combined list.
  const std::size_t localCellsBegin = std::distance
                                      (cells.begin(), std::lower_bound(cells.begin(), cells.end(),
                                          std::make_pair(localCodesBegin,
                                              std::size_t(0))));
  assert(localCellsBegin == cells.size() ||
         (cells[localCellsBegin].first >= localCodesBegin &&
          cells[localCellsBegin].first <= _cellCodes.back()));
  const std::size_t localCellsEnd = std::distance
                                    (cells.begin(),
                                     std::lower_bound(cells.begin(), cells.end(),
                                         std::make_pair(localCodesEnd, std::size_t(0))));
  assert(localCellsEnd == cells.size() ||
         (cells[localCellsEnd].first >= localCodesEnd &&
          cells[localCellsEnd].first <= _cellCodes.back()));

  // Rebuild the cell data structures.
  // The cell codes.
  _cellCodes.resize(cells.size() + 1);
  for (std::size_t i = 0; i != cells.size(); ++i) {
    _cellCodes[i] = cells[i].first;
  }
  _cellCodes.back() = morton.maxCode() + 1;
  // The cell delimiters.
  _cellDelimiters.resize(cells.size() + 1);
  _cellDelimiters[0] = 0;
  for (std::size_t i = 0; i != cells.size(); ++i) {
    _cellDelimiters[i + 1] = _cellDelimiters[i] + cells[i].second;
  }
  // The lookup table.
  buildLookupTable();

  // Fix the particles' data structures.
  // Insert placeholders for the new particles.
  particles.insert(particles.begin(), std::size_t(cellBegin(localCellsBegin)),
                   Particle());
  particles.insert(particles.end(), std::size_t(cellBegin(cellsSize()) -
                   cellBegin(localCellsEnd)),
                   Particle());
  assert(particles.size() == cellBegin(cellsSize()));
  // Insert NaN placeholders for the new starting positions.
  if (_padding != 0) {
    _startingPositions.insert(_startingPositions.begin(),
                              std::size_t(cellBegin(localCellsBegin)),
                              ext::filled_array<Point>
                              (std::numeric_limits<Float>::quiet_NaN()));
    _startingPositions.insert(_startingPositions.end(),
                              std::size_t(cellBegin(cellsSize()) -
                                          cellBegin(localCellsEnd)),
                              ext::filled_array<Point>
                              (std::numeric_limits<Float>::quiet_NaN()));
    assert(_startingPositions.size() == particles.size());
  }

  // Build the vector of adjacent cells.
  calculateAdjacentCells();

  // Return the local range of cells.
  return std::make_pair(localCellsBegin, localCellsEnd);
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
eraseShadow(const std::size_t localCellsBegin,
            const std::size_t localCellsEnd)
{
  // Check the range of local particles.
  assert(localCellsBegin <= localCellsEnd && localCellsEnd <= cellsSize());

  // Invalidate the data structures for the particles.
  adjacentCells.clear();
  _lookupTable.clear();
  _startingPositions.clear();

  // Erase the particles and their codes.
  const std::size_t localParticlesBegin = cellBegin(localCellsBegin);
  const std::size_t localParticlesEnd = cellBegin(localCellsEnd);
  assert(localParticlesBegin <= localParticlesEnd &&
         localParticlesEnd <= particles.size());
  particles.erase(particles.begin() + localParticlesEnd, particles.end());
  particles.erase(particles.begin(), particles.begin() + localParticlesBegin);

  // Erase the cells.
  assert(numAdjacentNeighbors.size() == cellsSize());
  numAdjacentNeighbors.erase(numAdjacentNeighbors.begin() + localCellsEnd,
                             numAdjacentNeighbors.end());
  numAdjacentNeighbors.erase(numAdjacentNeighbors.begin(),
                             numAdjacentNeighbors.begin() + localCellsBegin);
  // Note that there is one more delimiter than the number of cells.
  _cellDelimiters.erase(_cellDelimiters.begin() + localCellsEnd + 1,
                        _cellDelimiters.end());
  _cellDelimiters.erase(_cellDelimiters.begin(),
                        _cellDelimiters.begin() + localCellsBegin);
  _cellCodes.erase(_cellCodes.begin() + localCellsEnd, _cellCodes.end());
  _cellCodes.erase(_cellCodes.begin(), _cellCodes.begin() + localCellsBegin);
}


template<typename _Traits>
inline
std::size_t
MortonOrder<_Traits>::
_indexDirect(const Code code) const
{
  return _lookupTable(code);
}


template<typename _Traits>
inline
std::size_t
MortonOrder<_Traits>::
_indexForward(const Code code) const
{
  std::size_t i = _lookupTable(code);
  // Do a forward search to find the cell.
  while (_cellCodes[i] < code) {
    ++i;
  }
  return i;
}


template<typename _Traits>
inline
std::size_t
MortonOrder<_Traits>::
_indexBinary(const Code code) const
{
  // Do a binary search to find the cell. Using the lookup table to
  // narrow the search is much faster than using the whole array.
  const Code shift = Code(1) << _lookupTable.shift();
  const std::size_t begin = _lookupTable(code);
  return begin + std::lower_bound(&_cellCodes[begin],
                                  &_cellCodes[_lookupTable(code + shift)],
                                  code) - &_cellCodes[begin];
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
buildLookupTable()
{
  _timer.start();
  _lookupTable.initialize(_cellCodes, 4 * _cellCodes.size());
  _timer.stop();
  _timeBuildLookupTable += _timer.elapsed();
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
calculateAdjacentCells()
{
  _timer.start();
  // Clear the sequence of adjacent cells.
  adjacentCells.clear();

#ifdef _OPENMP
  typedef container::PackedArrayOfArrays<NeighborCell<Periodic> >
  PackedNeighbors;
  // Make a vector of packed arrays, one for each thread.
  std::vector<PackedNeighbors> adjacentForThread(omp_get_max_threads());
  #pragma omp parallel default(none) shared(adjacentForThread)
  {
    // Reference one of the packed arrays.
    PackedNeighbors* adjacent = &adjacentForThread[omp_get_thread_num()];

    std::vector<NeighborCell<Periodic> > adj;
    // Get our part of the cells.
    std::size_t begin, end;
    numerical::getPartitionRange(cellsSize(), &begin, &end);
    // Loop over the cells.
    for (std::size_t i = begin; i != end; ++i) {
      // Determine the particles in the adjacent cells.
      _findAdjacentCells(i, &adj);
      // Record the set of particles.
      adjacent->pushArray(adj.begin(), adj.end());
    }
  }
  // Merge the neighbors from the threads.
  adjacentCells.rebuild(adjacentForThread);
#else
  std::vector<NeighborCell<Periodic> > adj;
  // Loop over the cells.
  for (std::size_t i = 0; i != cellsSize(); ++i) {
    // Determine the particles in the adjacent cells.
    _findAdjacentCells(i, &adj);
    // Record the set of particles.
    adjacentCells.pushArray(adj.begin(), adj.end());
  }
  // Note that sorting each adjacency list would not improve performance.
  // There are long jumps in the space-filling curve at the cell boundaries
  // regardless.
#endif
  assert(adjacentCells.numArrays() == cellsSize());

  // Calculate the number of adjacent neighbors for each cell.
  numAdjacentNeighbors.resize(cellsSize());
  for (std::size_t i = 0; i != numAdjacentNeighbors.size(); ++i) {
    // The number of particles in the adjacent cells times the number of
    // particles in the cell.
    numAdjacentNeighbors[i] = countAdjacentParticles(i) *
                              (cellEnd(i) - cellBegin(i));
  }

  _timer.stop();
  _timeAdjacentCells += _timer.elapsed();
}


template<typename _Traits>
inline
typename MortonOrder<_Traits>::IndexRange
MortonOrder<_Traits>::
_adjacentRange(const std::size_t cell, std::false_type /*Periodic*/)
const
{
  // Convert the Morton code to discrete coordinates.
  const DiscretePoint coords = morton.coordinates(_cellCodes[cell]);
  // Determine the range of coordinates. Start with just the center.
  IndexRange range = {ext::filled_array <
                      std::array<std::size_t, Dimension> > (1),
                      ext::convert_array<std::size_t>(coords)
                     };
  // Extend if the adjacent cells are in the grid.
  for (std::size_t i = 0; i != coords.size(); ++i) {
    if (coords[i] != 0) {
      --range.bases[i];
      ++range.extents[i];
    }
    if (coords[i] != morton.cellExtents()[i] - 1) {
      ++range.extents[i];
    }
  }
  return range;
}


template<typename _Traits>
inline
typename MortonOrder<_Traits>::IndexRange
MortonOrder<_Traits>::
_adjacentRange(const std::size_t cell, std::true_type /*Periodic*/)
const
{
  // Convert the Morton code to discrete coordinates.
  const DiscretePoint coords = morton.coordinates(_cellCodes[cell]);
  // The offset range of coordinates.
  IndexRange range = {ext::filled_array <
                      std::array<std::size_t, Dimension> > (3),
                      ext::convert_array<std::size_t>(coords)
                     };
  return range;
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
_findAdjacentCells(const std::size_t cell,
                   std::vector<NeighborCell<false> >* adjacent) const
{
  typedef container::SimpleMultiIndexRangeIterator<Dimension> IndexIterator;

  adjacent->clear();
  // The range of cells.
  const IndexRange range = _adjacentRange(cell, std::false_type());
  // Loop over the adjacent cells.
  const IndexIterator End = IndexIterator::end(range);
  for (IndexIterator j = IndexIterator::begin(range); j != End; ++j) {
    // The code for this cell.
    const Code code = morton.code
                      (ext::convert_array<IntegerTypes::DiscreteCoordinate>(*j));
    const std::size_t k = index(code);
    // If the cell is non-empty.
    if (_cellCodes[k] == code) {
      adjacent->push_back(NeighborCell<false>{k});
    }
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
_findAdjacentCells(const std::size_t cell,
                   std::vector<NeighborCell<true> >* adjacent) const
{
  typedef container::SimpleMultiIndexRangeIterator<Dimension> IndexIterator;

  adjacent->clear();
  // The offset range of cells.
  const IndexRange offsetRange =
    _adjacentRange(cell, std::true_type());
  DiscretePoint coords;
  // Loop over the adjacent cells.
  const IndexIterator End = IndexIterator::end(offsetRange);
  for (IndexIterator j = IndexIterator::begin(offsetRange); j != End;
       ++j) {
    // Determine the cell coordinates and offset index from the offset
    // coordinates.
    coords = ext::convert_array<IntegerTypes::DiscreteCoordinate>(*j);
    std::size_t offsetIndex = 0;
    for (std::size_t d = 0, stride = 1; d != Dimension;
         ++d, stride *= 3) {
      if (coords[d] == 0) {
        coords[d] = morton.cellExtents()[d] - 1;
      }
      else if (coords[d] == morton.cellExtents()[d] + 1) {
        coords[d] = 0;
        offsetIndex += 2 * stride;
      }
      else {
        coords[d] -= 1;
        offsetIndex += stride;
      }
    }
    // The code for this cell.
    const Code code = morton.code(coords);
    const std::size_t k = index(code);
    // If the cell is non-empty.
    if (_cellCodes[k] == code) {
      adjacent->push_back(NeighborCell<true>{k, offsetIndex});
    }
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
_normalizePositions(std::true_type /*Periodic*/)
{
  Point x;
  // For each particle.
  for (std::size_t i = 0; i != particles.size(); ++i) {
    x = _getPosition(particles[i]);
    // For each dimension.
    for (std::size_t j = 0; j != Dimension; ++j) {
      x[j] -= floor((x[j] - morton.lowerCorner()[j]) / morton.lengths()[j]) *
              morton.lengths()[j];
    }
    _setPosition(&particles[i], x);
  }
}


template<typename _Traits>
inline
std::size_t
MortonOrder<_Traits>::
index(const Code code) const
{
  // Using a pointer to member function would not be any more efficient
  // than the branch below.
  if (_lookupTable.shift() == 0) {
    return _indexDirect(code);
  }
  // This threshold was determined with the indexOrderedF3 program in
  // test/performance/particle/serial.
  else if (_lookupTable.shift() <= 6) {
    return _indexForward(code);
  }
  else {
    return _indexBinary(code);
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
positionsInAdjacent(const std::size_t cell, std::vector<Point>* positions)
const
{
  positions->clear();
  for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& neighborCell = adjacentCells(cell, i);
    const Neighbor<Periodic> end = cellEnd(neighborCell);
    for (Neighbor<Periodic> neighbor = cellBegin(neighborCell);
         neighbor.particle != end.particle; ++neighbor) {
      positions->push_back(neighborPosition(neighbor));
    }
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
positionsInAdjacent(const std::vector<Point>& cachedPositions,
                    const std::size_t cell, std::vector<Point>* positions)
const
{
  positions->clear();
  for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& neighborCell = adjacentCells(cell, i);
    const Neighbor<Periodic> end = cellEnd(neighborCell);
    for (Neighbor<Periodic> neighbor = cellBegin(neighborCell);
         neighbor.particle != end.particle; ++neighbor) {
      positions->push_back(neighborPosition(cachedPositions, neighbor));
    }
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
positionsInAdjacent(const std::size_t cell, std::vector<std::size_t>* indices,
                    std::vector<Point>* positions) const
{
  indices->clear();
  positions->clear();
  for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& neighborCell = adjacentCells(cell, i);
    const Neighbor<Periodic> end = cellEnd(neighborCell);
    for (Neighbor<Periodic> neighbor = cellBegin(neighborCell);
         neighbor.particle != end.particle; ++neighbor) {
      indices->push_back(neighbor.particle);
      positions->push_back(neighborPosition(neighbor));
    }
  }
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
positionsInAdjacent(const std::size_t cell,
                    std::vector<Neighbor<Periodic> >* neighbors,
                    std::vector<Point>* positions) const
{
  neighbors->clear();
  positions->clear();
  for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& neighborCell = adjacentCells(cell, i);
    const Neighbor<Periodic> end = cellEnd(neighborCell);
    for (Neighbor<Periodic> neighbor = cellBegin(neighborCell);
         neighbor.particle != end.particle; ++neighbor) {
      neighbors->push_back(neighbor);
      positions->push_back(neighborPosition(neighbor));
    }
  }
}


template<typename _Traits>
inline
std::size_t
MortonOrder<_Traits>::
countAdjacentParticles(std::size_t cell) const
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
    const NeighborCell<Periodic>& adj = adjacentCells(cell, i);
    count += cellEnd(adj.cell) - cellBegin(adj.cell);
  }
  return count;
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
printMemoryUsageTable(std::ostream& out) const
{
  out << ",used,capacity\n"
      << "particles,"
      << particles.size() * sizeof(Particle) << ','
      << particles.capacity() * sizeof(Particle) << '\n'
      << "cellDelimiters,"
      << _cellDelimiters.size() * sizeof(std::size_t) << ','
      << _cellDelimiters.capacity() * sizeof(std::size_t) << '\n'
      << "cellCodes,"
      << _cellCodes.size() * sizeof(Code) << ','
      << _cellCodes.capacity() * sizeof(Code) << '\n'
      << "lookupTable,"
      << _lookupTable.memoryUsage() << ','
      << _lookupTable.memoryCapacity() << '\n'
      << "adjacentCells,"
      << adjacentCells.memoryUsage() << ','
      << adjacentCells.memoryCapacity() << '\n'
      << "numAdjacentNeighbors,"
      << numAdjacentNeighbors.size() * sizeof(std::size_t) << ','
      << numAdjacentNeighbors.capacity() * sizeof(std::size_t) << '\n';
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
printInfo(std::ostream& out) const
{
  out << "Dimension = " << Dimension << '\n'
      << "particles.size() = " << particles.size() << '\n'
      << "cell codes =";
  for (std::size_t i = 0; i != _cellCodes.size(); ++i) {
    out << ' ' << _cellCodes[i];
  }
  out << '\n';
}


template<typename _Traits>
inline
void
MortonOrder<_Traits>::
printPerformanceInfo(std::ostream& out) const
{
  out << "Dimension = " << Dimension << ", Periodic = " << Periodic << '\n'
      << "Lower corner = " << morton.lowerCorner() << '\n'
      << "Lengths = " << morton.lengths() << '\n'
      << "Cell lengths = " << morton.cellLengths() << '\n'
      << "Num levels = " << morton.numLevels() << '\n'
      << "Cell extents = " << morton.cellExtents() << '\n'
      << "Num cells capacity = " << morton.maxCode() + 1 << '\n'
      << "Interaction distance = " << interactionDistance() << '\n'
      << "Padding = " << padding()
      << ", fraction = " << padding() / interactionDistance() << '\n'
      << "Number of particles = " << particles.size() << '\n'
      << "Number of cells = " << cellsSize() << '\n'
      << "Particles per cell = " << double(particles.size()) / cellsSize()
      << '\n'
      << '\n'
      << "Reorder count = " << _reorderCount << '\n'
      << "Repair count = " << _repairCount << '\n';
  // Time totals.
  {
    const std::size_t Num = 4;
    const std::array<const char*, Num> names = {{
        "CheckOrder",
        "Order",
        "RecStartPos",
        "LookupTable"
      }
    };
    const std::array<double, Num> values = {
      {
        _timeIsOrderValid,
        _timeOrder,
        _timeRecordStartingPositions,
        _timeBuildLookupTable
      }
    };
    out << "\nTime totals:\n";
    // Column headers.
    for (std::size_t i = 0; i != names.size(); ++i) {
      out << names[i];
      if (i != names.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
    // Values.
    for (std::size_t i = 0; i != values.size(); ++i) {
      out << values[i];
      if (i != values.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
  }
  // Time per operation.
  {
    const std::size_t Num = 4;
    const std::array<const char*, Num> names = {{
        "CheckOrder",
        "Order",
        "RecStartPos",
        "LookupTable"
      }
    };
    const std::array<double, Num> values = {
      {
        _timeIsOrderValid / _repairCount,
        _timeOrder / _reorderCount,
        _timeRecordStartingPositions / _reorderCount,
        _timeBuildLookupTable / _reorderCount
      }
    };
    out << "\nTime per operation:\n";
    // Column headers.
    for (std::size_t i = 0; i != names.size(); ++i) {
      out << names[i];
      if (i != names.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
    // Values.
    for (std::size_t i = 0; i != values.size(); ++i) {
      out << values[i];
      if (i != values.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
  }
  out << "\nMemory Usage:\n";
  printMemoryUsageTable(out);
}


} // namespace particle
}

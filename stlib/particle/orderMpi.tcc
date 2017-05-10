// -*- C++ -*-

#if !defined(__particle_orderMpi_tcc__)
#error This file is an implementation detail of orderMpi.
#endif

namespace stlib
{
namespace particle
{


template<typename _Traits>
inline
MortonOrderMpi<_Traits>::
MortonOrderMpi(MPI_Comm comm,
               const geom::BBox<Float, Dimension>& domain,
               const Float interactionDistance, const Float shadowWidth,
               const Float padding) :
  Base(domain, interactionDistance, padding),
  interior(),
  exchanged(),
  shadow(),
  maxLoadImbalance(0.01),
  _comm(),
  _delimiters(mpi::commSize(comm) + 1),
  _totalNumAdjacentNeighbors(std::numeric_limits<std::size_t>::max()),
  _mpiCodeType(),
  _mpiSizeType(),
  _mpiFloatType(),
  _shadowWidth(shadowWidth),
  _shadowIndexOffsets(),
  _isExchangePatternDefined(false),
  _localCellsBegin(0),
  _localCellsEnd(0),
  _receiving(),
  _processCellLists(),
  _sendProcessIds(),
  _particleSendBuffers(),
  _startingImbalance(0),
  _partitionCount(0),
  _reorderCount(0),
  _repairCount(0),
  _timer(),
  _timeReorder(0),
  _timePartition(0),
  _timeDistributeUnordered(0),
  _numDistributeSent(0),
  _timeBuildExchangePattern(0),
  _timeExchangePost(0),
  _timeExchangeWait(0),
  _numNeighborsSend(0),
  _numNeighborsReceive(0),
  _numExchangeSent(0),
  _numExchangeReceived(0)
{
  MPI_Comm_dup(comm, &_comm);
  _defineMpiTypes();
  _buildShadowIndexOffsets();
  _checkGeometry();
}


template<typename _Traits>
inline
MortonOrderMpi<_Traits>::
MortonOrderMpi(MPI_Comm comm) :
  Base(),
  interior(),
  exchanged(),
  shadow(),
  maxLoadImbalance(0.01),
  _comm(),
  _delimiters(mpi::commSize(comm) + 1),
  _totalNumAdjacentNeighbors(std::numeric_limits<std::size_t>::max()),
  _mpiCodeType(),
  _mpiSizeType(),
  _mpiFloatType(),
  _shadowWidth(0),
  _shadowIndexOffsets(),
  _isExchangePatternDefined(false),
  _localCellsBegin(0),
  _localCellsEnd(0),
  _receiving(),
  _processCellLists(),
  _sendProcessIds(),
  _particleSendBuffers(),
  _startingImbalance(0),
  _partitionCount(0),
  _reorderCount(0),
  _repairCount(0),
  _timer(),
  _timeReorder(0),
  _timePartition(0),
  _timeDistributeUnordered(0),
  _numDistributeSent(0),
  _timeBuildExchangePattern(0),
  _timeExchangePost(0),
  _timeExchangeWait(0),
  _numNeighborsSend(0),
  _numNeighborsReceive(0),
  _numExchangeSent(0),
  _numExchangeReceived(0)
{
  MPI_Comm_dup(comm, &_comm);
  _defineMpiTypes();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
checkValidity() const
{
#ifndef NDEBUG
  assert(maxLoadImbalance > 0);
  // Check the sets of interior, exchanged, and shadow particles.
  assert(interior.size() + exchanged.size() + shadow.size() ==
         Base::cellsSize());
  assert(interior.size() + exchanged.size() == localCellsSize());
  // Check the local range of cells.
  assert(_localCellsBegin <= Base::cellsSize());
  assert(_localCellsBegin <= _localCellsEnd);
  assert(_localCellsEnd <= Base::cellsSize());
  // Check that all of the codes are within the range specified by the
  // delimiters.
  {
    const int rank = mpi::commRank(_comm);
    const Code lower = _delimiters[rank];
    const Code upper = _delimiters[rank + 1];
    if (_isExchangePatternDefined) {
      for (std::size_t i = 0; i != _localCellsBegin; ++i) {
        assert(Base::_cellCodes[i] < lower);
        // CONTINUE HERE
      }
      for (std::size_t i = _localCellsBegin; i != _localCellsEnd; ++i) {
        assert(lower <= Base::_cellCodes[i] && Base::_cellCodes[i] < upper);
      }
      for (std::size_t i = _localCellsEnd; i != Base::cellsSize(); ++i) {
        assert(Base::_cellCodes[i] >= upper);
        // CONTINUE HERE
      }
    }
    else {
      for (std::size_t i = 0; i != Base::cellsSize(); ++i) {
        assert(lower <= Base::_cellCodes[i] && Base::_cellCodes[i] < upper);
      }
    }
  }
  assert(Base::_cellCodes.back() == Base::morton.maxCode() + 1);
#endif
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_defineMpiTypes()
{
  if (sizeof(Code) == sizeof(unsigned long long)) {
    _mpiCodeType = MPI_UNSIGNED_LONG_LONG;
  }
  else if (sizeof(Code) == sizeof(unsigned long)) {
    _mpiCodeType = MPI_UNSIGNED_LONG;
  }
  else if (sizeof(Code) == sizeof(unsigned)) {
    _mpiCodeType = MPI_UNSIGNED;
  }
  else if (sizeof(Code) == sizeof(unsigned short)) {
    _mpiCodeType = MPI_UNSIGNED_SHORT;
  }
  else if (sizeof(Code) == sizeof(unsigned char)) {
    _mpiCodeType = MPI_UNSIGNED_CHAR;
  }
  else {
    assert(false);
  }

  if (sizeof(std::size_t) == sizeof(unsigned long long)) {
    _mpiSizeType = MPI_UNSIGNED_LONG_LONG;
  }
  else if (sizeof(std::size_t) == sizeof(unsigned long)) {
    _mpiSizeType = MPI_UNSIGNED_LONG;
  }
  else {
    assert(false);
  }

  if (sizeof(Float) == sizeof(float)) {
    _mpiFloatType = MPI_FLOAT;
  }
  else if (sizeof(Float) == sizeof(double)) {
    _mpiFloatType = MPI_DOUBLE;
  }
  else {
    assert(false);
  }
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_buildShadowIndexOffsets()
{
  typedef container::MultiIndexRange<Dimension> IndexRange;
  typedef container::MultiIndexRangeIterator<Dimension> IndexIterator;
  typedef typename IndexRange::SizeList SizeList;
  typedef typename IndexRange::IndexList IndexList;

  assert(_shadowWidth >= Base::interactionDistance());
  // Calculate the number of cells to include in each direction. Note that
  // we allow for round-off errors for the case that the shadow width is
  // an integer multiple of the interaction distance.
  const std::ptrdiff_t depth =
    std::ptrdiff_t(std::ceil(_shadowWidth / Base::interactionDistance() *
                             (1 - 2 *
                              std::numeric_limits<Float>::epsilon())));
  if (Periodic) {
    // Check that the cell extents are compatible with the depth.
    // The index offsets should not overlap.
    for (std::size_t j = 0; j != Dimension; ++j) {
      assert(DiscreteCoordinate(1 + 2 * depth) <=
             Base::morton.cellExtents()[j]);
    }
  }
  // Make a range that extends from -depth to depth in each dimension.
  const IndexRange range(ext::filled_array<SizeList>(2 * depth + 1),
                         ext::filled_array<IndexList>(-depth));
  const Point cellLengths = Base::morton.cellLengths();
  // Allow for motion of the local particles and the shadow particles of
  // up to half of the padding each. Work with the squared distance.
  const Float threshold = (_shadowWidth + Base::padding()) *
                          (_shadowWidth + Base::padding());
  const IndexList Zero = ext::filled_array<IndexList>(0);
  // Loop over the cells in the cube.
  const IndexIterator end = IndexIterator::end(range);
  for (IndexIterator index = IndexIterator::begin(range); index != end;
       ++index) {
    // Do not include the center cell itself.
    if (*index == Zero) {
      continue;
    }
    // Get a lower bound on the distance between particles in the cells by
    // calculating the distance between the cells. Specifically, use the
    // difference between the closest corners.
    Float dist = 0;
    for (std::size_t i = 0; i != Dimension; ++i) {
      Float diff = std::abs((*index)[i]);
      if (diff > 0) {
        --diff;
      }
      dist += (diff * cellLengths[i]) * (diff * cellLengths[i]);
    }
    // If the particles could be within the shadow width of each other.
    if (dist < threshold) {
      // Record the index offset for the cell.
      _shadowIndexOffsets.push_back(*index);
    }
  }
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_checkGeometry()
{
  // Pack the description of the geometry into an array of float's.
  std::array < Float, 2 * Dimension + 3 > geometry;
  std::size_t n = 0;
  for (std::size_t i = 0; i != Dimension; ++i) {
    geometry[n++] = Base::lowerCorner()[i];
  }
  for (std::size_t i = 0; i != Dimension; ++i) {
    geometry[n++] = Base::lengths()[i];
  }
  geometry[n++] = Base::interactionDistance();
  geometry[n++] = Base::padding();
  geometry[n++] = _shadowWidth;
  assert(n == geometry.size());

  std::array < Float, 2 * Dimension + 3 > buffer = geometry;
  mpi::bcast(&buffer[0], buffer.size(), _comm);
  assert(buffer == geometry);
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_neighborCellCodes(const std::size_t cellIndex,
                   std::vector<Code>* neighborCodes,
                   std::false_type /*Periodic*/) const
{
  // Work with signed integers.
  typedef std::array<std::ptrdiff_t, Dimension> SignedDiscretePoint;

  // Convert the Morton code for the center cell to discrete coordinates.
  const SignedDiscretePoint center = ext::convert_array<std::ptrdiff_t>
                                     (Base::morton.coordinates(Base::_cellCodes[cellIndex]));
  const SignedDiscretePoint cellExtents = ext::convert_array<std::ptrdiff_t>
                                          (Base::morton.cellExtents());

  // Make a list of the neighbor cell codes.
  neighborCodes->clear();
  neighborCodes->reserve(_shadowIndexOffsets.size());
  SignedDiscretePoint coords;
  // Loop over the index offsets.
  for (std::size_t i = 0; i != _shadowIndexOffsets.size(); ++i) {
    coords = center + _shadowIndexOffsets[i];
    bool valid = true;
    for (std::size_t j = 0; j != Dimension; ++j) {
      // Check if the cell is in the grid.
      if (coords[j] < 0 || coords[j] >= cellExtents[j]) {
        valid = false;
      }
    }
    if (valid) {
      // Record the code for this neighbor cell.
      neighborCodes->push_back
      (Base::morton.code(ext::convert_array<DiscreteCoordinate>(coords)));
    }
  }
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_neighborCellCodes(const std::size_t cellIndex,
                   std::vector<Code>* neighborCodes,
                   std::true_type /*Periodic*/) const
{
  // Convert the Morton code for the center cell to discrete coordinates.
  const DiscretePoint center =
    Base::morton.coordinates(Base::_cellCodes[cellIndex]);

  // Make a list of the neighbor cell codes.
  neighborCodes->resize(_shadowIndexOffsets.size());
  DiscretePoint coords;
  // Loop over the index offsets.
  for (std::size_t i = 0; i != _shadowIndexOffsets.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      coords[j] = (center[j] + Base::morton.cellExtents()[j] +
                   _shadowIndexOffsets[i][j]) %
                  Base::morton.cellExtents()[j];
    }
    // Record the code for this neighbor cell.
    (*neighborCodes)[i] = Base::morton.code(coords);
  }
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_sendToProcessList(const std::size_t cellIndex,
                   std::vector<std::size_t>* processes) const
{
  // Get the codes for the neighboring cells.
  std::vector<Code> neighborCodes;
  _neighborCellCodes(cellIndex, &neighborCodes,
                     std::integral_constant<bool, Periodic>());
  // Continue processing with the neighbor codes.
  _sendToProcessList(&neighborCodes, processes);
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_sendToProcessList(std::vector<Code>* neighborCodes,
                   std::vector<std::size_t>* processes) const
{
  processes->clear();
  // Handle the trivial case.
  if (neighborCodes->empty()) {
    return;
  }
  const std::size_t commRank = mpi::commRank(_comm);
  // Sort the list of neighboring cell codes.
  std::sort(neighborCodes->begin(), neighborCodes->end());
  // Start with an invalid upper delimiter for the current process.
  Code upper = 0;
  // Loop over the neighboring cells.
  for (std::size_t i = 0; i != neighborCodes->size(); ++i) {
    // If we have moved to a new process.
    if ((*neighborCodes)[i] >= upper) {
      // Determine the process.
      const std::size_t p = _process((*neighborCodes)[i]);
      // The upper delimiter.
      upper = _delimiters[p + 1];
      // Record the process.
      if (p != commRank) {
        processes->push_back(p);
      }
    }
  }
}


#if 0
template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_isValidShadow(const std::size_t cellIndex) const
{
  // Get the processes to which this cell should be sent.
  std::vector<std::size_t> processes;
  _sendToProcessList(cellIndex, &processes,
                     std::integral_constant<bool, Periodic>());
  CONTINUE;
}
#endif


template<typename _Traits>
template<typename _InputIterator>
inline
void
MortonOrderMpi<_Traits>::
setParticles(_InputIterator begin, _InputIterator end)
{
  // Set the particles in the local data structure.
  Base::setParticles(begin, end);

  // Initially, partition by particles.
  _partitionByParticles();
  // Distribute the particles.
  _distributePattern();
  // Exchange particles so that we can calculate the neighbors.
  exchangeParticles();

  // Then partition by neighbors.
  _partitionByNeighbors();
  // Distribute the particles. Find the neighbors.
  _distributePattern();
}


template<typename _Traits>
inline
bool
MortonOrderMpi<_Traits>::
repair()
{
  ++_repairCount;
  if (! isOrderValid()) {
    reorder();
    // Determine the exchange pattern.
    _buildExchangePattern();
    // Balance the load if necessary. (Ignore the return value.)
    _balanceLoad();
    return true;
  }
  return false;
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
exchangeParticles()
{
  //
  // Post all of the receives.
  //
  _timer.start();
  _numNeighborsReceive += _receiving.size();
  std::vector<MPI_Request> receiveRequests(_receiving.size());
  for (std::size_t i = 0; i != receiveRequests.size(); ++i) {
    _numExchangeReceived += _receiving[i].size;
    receiveRequests[i] =
      mpi::iRecv(&Base::particles[_receiving[i].position],
                 _receiving[i].size * sizeof(Particle),
                 MPI_BYTE, _receiving[i].source, EpParticleTag, _comm);
  }

  // Using threading dramatically hurts performance.
  _numNeighborsSend += _sendProcessIds.size();
  //
  // Make the send buffers and send each package.
  //
  std::vector<MPI_Request> sendRequests(_sendProcessIds.size());
  // For each process to which we send particles.
  for (std::size_t i = 0; i != _sendProcessIds.size(); ++i) {
    const std::size_t id = _sendProcessIds[i];
    typename container::PackedArrayOfArrays<Particle>::iterator buffer =
      _particleSendBuffers.begin(i);
    // For each cell that we will send.
    for (std::size_t j = 0; j != _processCellLists.size(id); ++j) {
      // Add the contents of the cell to the buffer.
      const std::size_t cell = _processCellLists(id, j);
      for (std::size_t k = Base::cellBegin(cell); k != Base::cellEnd(cell);
           ++k) {
        *buffer++ = Base::particles[k];
      }
    }
    _numExchangeSent += _particleSendBuffers.size(i);
    // Send the buffer.
    sendRequests[i] =
      mpi::iSend(&*_particleSendBuffers.begin(i),
                 _particleSendBuffers.size(i) * sizeof(Particle),
                 MPI_BYTE, id, EpParticleTag, _comm);
  }
  _timer.stop();
  _timeExchangePost += _timer.elapsed();

  //
  // Wait for the receives and sends to complete.
  //
  _timer.start();
  mpi::waitAll(&receiveRequests);
  mpi::waitAll(&sendRequests);
  _timer.stop();
  _timeExchangeWait += _timer.elapsed();
}


template<typename _Traits>
inline
bool
MortonOrderMpi<_Traits>::
isOrderValid() const
{
  // Record whether the local data is valid.
  bool const localValid = Base::isOrderValid(localParticlesBegin(),
                                             localParticlesEnd());
  // Perform a reduction with logical and then a broadcast.
  return mpi::allReduce(int(localValid), MPI_LAND, _comm);
}


// No longer used.
template<typename _Traits>
inline
bool
MortonOrderMpi<_Traits>::
isParticleLoadBalanced()
{
  // Gather the number of particles for each process to the root.
  std::size_t localSize = localParticlesEnd() - localParticlesBegin();
  std::vector<std::size_t> const sizes = mpi::gather(localSize, _comm);

  // Every time we check the load, decrease the starting imbalance by
  // a small amount. This is because the problem may admit better load
  // balancing as the simulation progresses. For example, the distribution
  // of particles may become more uniform.
  _startingImbalance *= 0.99;

  // Check for processes that have a load that exceeds the average by
  // more than the tolerance.
  bool isBalanced;
  if (mpi::commRank(_comm) == 0) {
    isBalanced =
      ext::max(sizes) <= std::size_t(Float(ext::sum(sizes)) / sizes.size()
                                     * (1 + std::max(maxLoadImbalance,
                                                     2 * _startingImbalance)));
  }

  // Broadcast the result to all processes.
  mpi::bcast(&isBalanced, _comm);

  return isBalanced;
}


template<typename _Traits>
inline
bool
MortonOrderMpi<_Traits>::
isNeighborsLoadBalanced()
{
  // Gather the number of particles for each process to the root.
  // For the root process, allocate memory for the results. For the rest,
  // just allocate one element so that &loads[0] is valid.
  assert(_totalNumAdjacentNeighbors !=
         std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> const loads =
    mpi::gather(_totalNumAdjacentNeighbors, _comm);

  // Every time we check the load, decrease the starting imbalance by
  // a small amount. This is because the problem may admit better load
  // balancing as the simulation progresses. For example, the distribution
  // of particles may become more uniform.
  _startingImbalance *= 0.99;

  // Check for processes that have a load that exceeds the average by
  // more than the tolerance.
  bool isBalanced;
  if (mpi::commRank(_comm) == 0) {
    isBalanced =
      ext::max(loads) <= std::size_t(Float(ext::sum(loads)) / loads.size()
                                     * (1 + std::max(maxLoadImbalance,
                                                     2 * _startingImbalance)));
  }

  // Broadcast the result to all processes.
  mpi::bcast(&isBalanced, _comm);

  return isBalanced;
}

template<typename _Traits>
inline
bool
MortonOrderMpi<_Traits>::
_balanceLoad()
{
  if (! isNeighborsLoadBalanced()) {
    // Determine the partitioning.
    _partitionByNeighbors();
    // Distribute the particles. Determine the exchange pattern.
    _distributePattern();
    return true;
  }
  return false;
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
reorder()
{
  // Start the timer.
  _timer.start();

  ++_reorderCount;

  // Get rid of the current exchange pattern.
  _clearExchangePattern();

  // Recalculate the codes and order the particles.
  Base::order();

  //
  // Determine the cells that should be sent to other processes.
  //
  const std::size_t commRank = mpi::commRank(_comm);
  const std::size_t commSize = mpi::commSize(_comm);

  std::vector<std::size_t> sendCellDelimiters(_delimiters.size());
  for (std::size_t i = 0; i != sendCellDelimiters.size(); ++i) {
    sendCellDelimiters[i] = Base::index(_delimiters[i]);
  }
  assert(sendCellDelimiters[0] == 0);
  assert(sendCellDelimiters.back() == Base::cellsSize());
  // The number of shadow particles that precede the local ones.
  const std::size_t lowerCount = Base::cellBegin(sendCellDelimiters[commRank]);
  // The number in this process.
  const std::size_t sameCount =
    Base::cellBegin(sendCellDelimiters[commRank + 1]) - lowerCount;
  //const std::size_t higherCount = Base::cellsSize() - sameCount - lowerCount;

  // Convert code delimiters to cell delimiters.
  std::vector<std::size_t> cellDelimiters(_delimiters.size());
  cellDelimiters[0] = 0;
  for (std::size_t i = 1; i != cellDelimiters.size(); ++i) {
    // Avoid index() calls if possible by checking for the case that we
    // do not have any cells that belong to the ith process.
    if (Base::_cellCodes[cellDelimiters[i - 1]] >= _delimiters[i]) {
      cellDelimiters[i] = cellDelimiters[i - 1];
    }
    else {
      cellDelimiters[i] = Base::index(_delimiters[i]);
    }
  }

  //
  // Gather and broadcast the migration information.
  //
  // Make an array of the particle send counts.
  std::vector<std::size_t> sendCounts(commSize);
  for (std::size_t i = 0; i != commSize; ++i) {
    sendCounts[i] = Base::cellBegin(cellDelimiters[i + 1]) -
                    Base::cellBegin(cellDelimiters[i]);
  }
  // We don't send particles to ourself.
  sendCounts[commRank] = 0;
  // Perform a reduction to the root process using the sum.
  std::vector<std::size_t> reducedCounts;
  mpi::reduce(sendCounts, &reducedCounts, MPI_SUM, _comm);
  // Scatter so that each process knows how many particles it will receive.
  std::size_t receiveCount;
  MPI_Scatter(&reducedCounts[0], 1, MPI_LONG, &receiveCount, 1,
              MPI_LONG, 0, _comm);

  //
  // Send and receive particles.
  //
  // Initiate all of the sends.
  std::vector<MPI_Request> requests;
  for (std::size_t i = 0; i != commSize; ++i) {
    if (sendCounts[i] != 0) {
      const std::size_t firstParticle = Base::cellBegin(cellDelimiters[i]);
      const std::size_t numParticles =
        Base::cellBegin(cellDelimiters[i + 1]) - firstParticle;
      requests.push_back(mpi::iSend(&Base::particles[firstParticle],
                                    numParticles * sizeof(Particle),
                                    MPI_BYTE, i, ReorderParticleTag, _comm));
    }
  }
  // Initiate receives until we have received all of the particles.
  std::vector<Particle> buffer(receiveCount);
  {
    MPI_Status status;
    std::size_t i = 0;
    while (i != buffer.size()) {
      MPI_Recv(&buffer[i], (buffer.size() - i) * sizeof(Particle),
               MPI_BYTE, MPI_ANY_SOURCE, ReorderParticleTag, _comm, &status);
      i += mpi::getCount(status, MPI_BYTE) / sizeof(Particle);
    }
  }
  // Wait for the sends to complete.
  for (std::size_t i = 0; i != requests.size(); ++i) {
    mpi::wait(&requests[i]);
  }

  // Remove the particles that were sent away.
  Base::particles.erase(Base::particles.begin(),
                        Base::particles.begin() + lowerCount);
  Base::particles.erase(Base::particles.begin() + sameCount,
                        Base::particles.end());
  // Merge the particles that were received.
  Base::particles.insert(Base::particles.end(), buffer.begin(), buffer.end());
  // Note that we haven't updated the codes or the starting positions. We will
  // take care of that below.

  // Reorder the local particles.
  Base::reorder();

  // Record the elapsed time.
  _timer.stop();
  _timeReorder += _timer.elapsed();
}


// Perform a binary reduction of the tables of cell data. Return the number of
// levels the table was shifted.
template<typename _Traits>
template<typename _T>
inline
std::size_t
MortonOrderMpi<_Traits>::
_reduce(std::vector<std::pair<Code, _T> >* table,
        const std::size_t maxBufferSize) const
{
  const std::size_t commSize = mpi::commSize(_comm);
  const std::size_t commRank = mpi::commRank(_comm);
  const std::size_t PairSize = sizeof(std::pair<Code, _T>);

  // A power of two that is at least the number of processes.
  std::size_t n = 1;
  while (n < commSize) {
    n *= 2;
  }

  std::vector<std::pair<Code, _T> > buffer(maxBufferSize);
  std::vector<std::pair<Code, _T> > received;

  {
    int size;
    MPI_Type_size(MPI_LONG, &size);
    assert(sizeof(std::size_t) == size);
  }
  // Binary reduction.
  std::size_t level = 0;
  for (; n > 1; n /= 2) {
    // If this process is a receiver.
    if (commRank < n / 2) {
      // If there is a sender.
      if (commRank + n / 2 < commSize) {
        MPI_Status status;
        // Get the shift level for the received table.
        std::size_t receivedLevel = 0;
        MPI_Recv(&receivedLevel, 1, _mpiSizeType, commRank + n / 2,
                 PartitionShiftTag, _comm, &status);
        assert(mpi::getCount(status, _mpiSizeType) == 1);
        // Get the frequency table.
        MPI_Recv(&buffer[0], PairSize * maxBufferSize, MPI_BYTE,
                 commRank + n / 2, ReduceTableTag, _comm, &status);
        const std::size_t recvCount = mpi::getCount(status, MPI_BYTE) /
                                      PairSize;
        // Make the received table from the buffer.
        received.clear();
        received.insert(received.end(), buffer.begin(),
                        buffer.begin() + recvCount);
        // Adjust the tables so they have the same shift level.
        if (level > receivedLevel) {
          shift<Dimension>(&received, level - receivedLevel);
        }
        else if (receivedLevel > level) {
          shift<Dimension>(table, receivedLevel - level);
          level = receivedLevel;
        }
        // Merge the received frequency table into this table.
        merge(table, received);
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      // Compress the table if necessary by right-shifting the codes.
      while (table->size() > maxBufferSize) {
        shift<Dimension>(table, 1);
        ++level;
      }
      // Send the shift level.
      MPI_Send(&level, 1, _mpiSizeType, commRank - n / 2,
               PartitionShiftTag, _comm);
      // Send the frequency table.
      MPI_Send(&(*table)[0], PairSize * table->size(), MPI_BYTE,
               commRank - n / 2, ReduceTableTag, _comm);
    }
  }
  return level;
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_partitionByParticles(const Float accuracyGoal)
{
  // Start the timer.
  _timer.start();
  // Get rid of the current exchange pattern.
  _clearExchangePattern();
  // Count the number of particles per cell.
  std::vector<std::pair<Code, std::size_t> > costs(Base::cellsSize());
  for (std::size_t i = 0; i != costs.size(); ++i) {
    costs[i] = std::make_pair(Base::_cellCodes[i],
                              Base::cellEnd(i) - Base::cellBegin(i));
  }
  // Partition using these costs.
  _partition(&costs, accuracyGoal);
  // Record the elapsed time.
  _timer.stop();
  _timePartition += _timer.elapsed();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_partitionByNeighbors(const Float accuracyGoal)
{
  // Start the timer.
  _timer.start();
  // Get rid of the current exchange pattern.
  _clearExchangePattern();
  // Count the number of potential neighbors per cell.
  std::vector<std::pair<Code, std::size_t> > costs(Base::cellsSize());
  assert(Base::numAdjacentNeighbors.size() == costs.size());
  for (std::size_t i = 0; i != costs.size(); ++i) {
    costs[i] = std::make_pair(Base::_cellCodes[i],
                              Base::numAdjacentNeighbors[i]);
  }
  // Partition using these costs.
  _partition(&costs, accuracyGoal);
  // Record the elapsed time.
  _timer.stop();
  _timePartition += _timer.elapsed();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_partition(std::vector<std::pair<Code, std::size_t> >* costs,
           const Float accuracyGoal)
{
  const std::size_t commSize = mpi::commSize(_comm);
  const std::size_t commRank = mpi::commRank(_comm);

  ++_partitionCount;

  // Determine an appropriate buffer size.
  const std::size_t maxBufferSize = std::size_t(commSize / accuracyGoal);
  // Binary reduction on the cell data.
  const std::size_t level = _reduce(costs, maxBufferSize);

  // Partition the particles using the combined frequency table.
  if (commRank == 0) {
    // Restore the codes to the original level by left-shifting them.
    if (level != 0) {
      for (std::size_t i = 0; i != costs->size(); ++i) {
        (*costs)[i].first <<= Dimension * level;
      }
    }
    // Copy the frequencies into a vector of weights.
    std::vector<std::size_t> weights(costs->size());
    for (std::size_t i = 0; i != weights.size(); ++i) {
      weights[i] = (*costs)[i].second;
    }
    // Partition the weights.
    std::vector<std::size_t> indexDelimiters;
    numerical::computePartitions(weights, commSize,
                                 std::back_inserter(indexDelimiters));
#if 0
    // CONTINUE REMOVE
    std::cerr << "costs:"
              << *costs
              << "weights:"
              << weights
              << "indexDelimiters:"
              << indexDelimiters;
#endif
    //
    // Compute the load imbalance.
    //
    // Determine the maximum weight for a process.
    std::size_t maxWeight = 0;
    for (std::size_t i = 0; i != commSize; ++i) {
      std::size_t w = 0;
      for (std::size_t j = indexDelimiters[i]; j != indexDelimiters[i + 1];
           ++j) {
        w += weights[j];
      }
      if (w > maxWeight) {
        maxWeight = w;
      }
    }
    // Calculate the imbalance using the maximum amount over the average.
    // Note that this is only stored on the root process.
    _startingImbalance = double(maxWeight * commSize) / ext::sum(weights) - 1;

    // Add an empty guard element. This is necessary because some of the
    // trailing parts may be empty. This way all of the index delimiters 
    // below will valid indices into *costs.
    costs->push_back(std::make_pair(Base::morton.maxCode() + 1,
                                    std::size_t(0)));
    // Convert the index delimiters to code delimiters.
    _delimiters.front() = 0;
    _delimiters.back() = Base::morton.maxCode() + 1;
    for (std::size_t i = 1; i < _delimiters.size() - 1; ++i) {
      _delimiters[i] = (*costs)[indexDelimiters[i]].first;
    }
    assert(std::is_sorted(_delimiters.begin(), _delimiters.end()));
  }
  // Broadcast the partitioning to all processes.
  mpi::bcast(&_delimiters, _comm);
}


// Distribute particles and determine the exchange pattern.
template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_distributePattern()
{
  // The exchange pattern must have been cleared.
  assert(! _isExchangePatternDefined);
  // We must be holding only local particles.
  assert(localParticlesEnd() - localParticlesBegin() ==
         Base::particles.size());
  // Calculate the codes. Record the starting positions.
  // Build the lookup table.
  Base::reorder();
  // Distribute the particles.
  _distributeUnordered();
  // Determine the exchange pattern.
  _buildExchangePattern();
}


// Distribute the particles according to the partition defined by the code
// delimiters.
template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_distributeUnordered()
{
  // Start the timer.
  _timer.start();

  // Ensure that we are not holding foreign particles.
  assert(! _isExchangePatternDefined);

  const std::size_t commSize = mpi::commSize(_comm);
  const std::size_t commRank = mpi::commRank(_comm);
  std::vector<Particle> newParticles, buffer;

  // Add the particles from this process.
  std::size_t begin = Base::cellBegin(Base::index(_delimiters[commRank]));
  std::size_t end = Base::cellBegin(Base::index(_delimiters[commRank + 1]));
  newParticles.insert(newParticles.end(),
                      Base::particles.begin() + begin,
                      Base::particles.begin() + end);
  // CONTINUE: This is slow. Use an MPI all-to-all instead.
  // Add the particles from each of the other processes.
  std::size_t source, target, sendCount, receiveCount;
  MPI_Request request;
  for (std::size_t offset = 1; offset != commSize; ++offset) {
    // The processes to receive from and send to.
    source = (commRank + commSize - offset) % commSize;
    target = (commRank + offset) % commSize;
    // The range of particles to send.
    begin = Base::cellBegin(Base::index(_delimiters[target]));
    end = Base::cellBegin(Base::index(_delimiters[target + 1]));
    // Sanity check.
    assert(end >= begin);
    sendCount = end - begin;
    _numDistributeSent += sendCount;
    // Send the number of particles that we will send.
    request = mpi::iSend(&sendCount, 1, target, DuCountTag, _comm);
    // Receive the number of particles that we will receive.
    mpi::recv(&receiveCount, 1, MPI_LONG, source, DuCountTag, _comm);
    // Wait for the send to complete.
    mpi::wait(&request);

    // Send the particles.
    if (sendCount != 0) {
      request = mpi::iSend(&Base::particles[begin],
                           sendCount * sizeof(Particle), MPI_BYTE,
                           target, DuParticleTag, _comm);
    }
    // Receive the particles
    if (receiveCount != 0) {
      buffer.resize(receiveCount);
      mpi::recv(&buffer[0], receiveCount * sizeof(Particle), MPI_BYTE,
                source, DuParticleTag, _comm);
      newParticles.insert(newParticles.end(), buffer.begin(), buffer.end());
    }
    // Wait for the send to complete.
    if (sendCount != 0) {
      mpi::wait(&request);
    }
  }

  Base::setParticles(newParticles.begin(), newParticles.end());

  // Record the elapsed time.
  _timer.stop();
  _timeDistributeUnordered += _timer.elapsed();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_buildExchangePattern()
{
  // Start the timer.
  _timer.start();

  const std::size_t commRank = mpi::commRank(_comm);
  const std::size_t commSize = mpi::commSize(_comm);

  // Ensure that the sets of interior, exchanged, and shadow cells have
  // been cleared.
  assert(interior.empty());
  assert(exchanged.empty());
  assert(shadow.empty());

  //
  // For each nonempty cell, determine the processes to send to.
  //
  std::vector<std::pair<std::size_t, std::size_t> > processCellPairs;
  {
    std::vector<std::size_t> processes;
    // For each nonempty cell.
    for (std::size_t i = 0; i != Base::cellsSize(); ++i) {
      // Record the processes to which we should send the cell.
      _sendToProcessList(i, &processes);
      if (processes.empty()) {
        interior.push_back(i);
      }
      else {
        exchanged.push_back(i);
      }
      for (std::size_t j = 0; j != processes.size(); ++j) {
        // Record the process/cell pair.
        processCellPairs.push_back(std::make_pair(processes[j], i));
      }
    }
  }
  // Sort by process.
  std::sort(processCellPairs.begin(), processCellPairs.end());

  //
  // For each process, determine the cells to send.
  //
  _processCellLists.clear();
  {
    std::size_t j = 0;
    for (std::size_t i = 0; i != commSize; ++i) {
      _processCellLists.pushArray();
      for (; j != processCellPairs.size() &&
           processCellPairs[j].first == i; ++j) {
        _processCellLists.push_back(processCellPairs[j].second);
      }
    }
  }

  //
  // Determine the processes to which we will send particles.
  //
  _sendProcessIds.clear();
  std::vector<std::size_t> willSend(commSize, 0);
  for (std::size_t i = 0; i != _processCellLists.numArrays(); ++i) {
    if (! _processCellLists.empty(i)) {
      _sendProcessIds.push_back(i);
      willSend[i] = 1;
    }
  }

  //
  // Determine the number of processes from which we will receive particles.
  //
  std::size_t numProcessesToReceive;
  {
    // Perform a reduction to the root process using the sum.
    std::vector<std::size_t> reducedCounts;
    mpi::reduce(willSend, &reducedCounts, MPI_SUM, _comm);
    // Scatter so that each process knows how many processes from which it
    // will receive particles.
    MPI_Scatter(&reducedCounts[0], 1, MPI_LONG, &numProcessesToReceive, 1,
                MPI_LONG, 0, _comm);
  }

  //
  // Calculate the number of particles to send to each process.
  //
  std::vector<std::size_t> sendParticleCounts(commSize, 0);
  for (std::size_t i = 0; i != sendParticleCounts.size(); ++i) {
    for (std::size_t j = 0; j != _processCellLists.size(i); ++j) {
      const std::size_t cell = _processCellLists(i, j);
      sendParticleCounts[i] += Base::cellEnd(cell) - Base::cellBegin(cell);
    }
  }

  //
  // Build the particle send buffers.
  //
  {
    std::vector<std::size_t> sizes(_sendProcessIds.size());
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      sizes[i] = sendParticleCounts[_sendProcessIds[i]];
    }
    _particleSendBuffers.rebuild(sizes.begin(), sizes.end());
  }

  {
    //
    // Determine the number of cells to receive from each process.
    //
    // Store the number of cells to send to each process.
    std::vector<std::size_t> sendCellCounts(commSize);
    for (std::size_t i = 0; i != sendCellCounts.size(); ++i) {
      sendCellCounts[i] = _processCellLists.size(i);
    }
    // Initiate sends.
    std::vector<MPI_Request> requests;
    for (std::size_t i = 0; i != sendCellCounts.size(); ++i) {
      if (sendCellCounts[i] != 0) {
        requests.push_back(mpi::iSend(&sendCellCounts[i], 1,
                                      MPI_LONG, i, BepCellCountTag, _comm));
      }
    }
    // Receive the counts.
    MPI_Status status;
    std::size_t c;
    std::vector<std::pair<std::size_t, std::size_t> > receiveCounts;
    for (std::size_t i = 0; i != numProcessesToReceive; ++i) {
      MPI_Recv(&c, 1, MPI_LONG, MPI_ANY_SOURCE, BepCellCountTag, _comm,
               &status);
      receiveCounts.push_back(std::make_pair
                              (std::size_t(status.MPI_SOURCE), c));
    }
    std::sort(receiveCounts.begin(), receiveCounts.end());
    // Wait for the sends to complete.
    mpi::waitAll(&requests);

    //
    // Get the information (codes and sizes) for cells that we will receive.
    //
    // Post all of the receives.
    std::vector<std::vector<std::pair<Code, std::size_t> > >
    codeSizeBuffers(receiveCounts.size());
    requests.clear();
    for (std::size_t i = 0; i != receiveCounts.size(); ++i) {
      codeSizeBuffers[i].resize(receiveCounts[i].second);
      requests.push_back(mpi::iRecv(&codeSizeBuffers[i][0],
                                    sizeof(std::pair<Code, std::size_t>) *
                                    codeSizeBuffers[i].size(), MPI_BYTE,
                                    receiveCounts[i].first,
                                    BepCodeSizeTag, _comm));
    }
    // Send each package.
    for (std::size_t i = 0; i != _processCellLists.numArrays(); ++i) {
      if (_processCellLists.empty(i)) {
        continue;
      }
      // Make the buffer of code/size pairs.
      std::vector<std::pair<Code, std::size_t> >
      buffer(_processCellLists.size(i));
      for (std::size_t j = 0; j != _processCellLists.size(i); ++j) {
        const std::size_t cell = _processCellLists(i, j);
        buffer[j] = std::make_pair(Base::_cellCodes[cell],
                                   Base::cellEnd(cell) -
                                   Base::cellBegin(cell));
      }
      // Send the buffer.
      mpi::send(&buffer[0],
                sizeof(std::pair<Code, std::size_t>) * buffer.size(),
                MPI_BYTE, i, BepCodeSizeTag, _comm);
    }
    // Wait for the receives to complete.
    for (std::size_t i = 0; i != requests.size(); ++i) {
      mpi::wait(&requests[i]);
    }

    // Build list of the cells (codes and sizes) that we will receive.
    std::vector<std::pair<Code, std::size_t> > codesSizes;
    for (std::size_t i = 0; i != codeSizeBuffers.size(); ++i) {
      codesSizes.insert(codesSizes.end(), codeSizeBuffers[i].begin(),
                        codeSizeBuffers[i].end());
    }

    // Rebuild the base class by inserting the cells.
    std::pair<std::size_t, std::size_t> localRange =
      Base::insertCells(codesSizes, _delimiters[commRank],
                        _delimiters[commRank + 1]);
    // Record the local range of cells and particles.
    _localCellsBegin = localRange.first;
    _localCellsEnd = localRange.second;

    //
    // Determine where to put the shadow particles.
    //
    _receiving.clear();
    {
      // Convert code delimiters to particle delimiters.
      std::vector<std::size_t> particleDelimiters(_delimiters.size());
      for (std::size_t i = 0; i != particleDelimiters.size(); ++i) {
        particleDelimiters[i] =
          Base::cellBegin(Base::index(_delimiters[i]));
      }
      assert(particleDelimiters.front() == 0);
      assert(particleDelimiters.back() == Base::particles.size());
      for (std::size_t i = 0; i != commSize; ++i) {
        // We don't receive particles from ourselves. Skip empty ranges.
        if (i == commRank ||
            particleDelimiters[i] == particleDelimiters[i + 1]) {
          continue;
        }
        _receiving.push_back(ReceivingInfo{
            i, particleDelimiters[i],
              particleDelimiters[i + 1] - particleDelimiters[i]
              });
      }
    }
  }

  // Correct the information for sending particles.
  _processCellLists += _localCellsBegin;

  // Correct the sets of interior and exchanged cells.
  for (std::size_t i = 0; i != interior.size(); ++i) {
    interior[i] += _localCellsBegin;
  }
  for (std::size_t i = 0; i != exchanged.size(); ++i) {
    exchanged[i] += _localCellsBegin;
  }
  // Build the set of shadow particles.
  shadow.resize(_localCellsBegin + Base::cellsSize() - _localCellsEnd);
  for (std::size_t i = 0; i != _localCellsBegin; ++i) {
    shadow[i] = i;
  }
  for (std::size_t i = _localCellsEnd; i != Base::cellsSize(); ++i) {
    shadow[_localCellsBegin + i - _localCellsEnd] = i;
  }

  _isExchangePatternDefined = true;

  // Calculate the total number of adjacent neighbors.
  _totalNumAdjacentNeighbors = 0;
  for (std::size_t i = _localCellsBegin; i != _localCellsEnd; ++i) {
    _totalNumAdjacentNeighbors += Base::numAdjacentNeighbors[i];
  }

  // Record the elapsed time.
  _timer.stop();
  _timeBuildExchangePattern += _timer.elapsed();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_clearExchangePattern()
{
  if (! _isExchangePatternDefined) {
    _localCellsBegin = 0;
    _localCellsEnd = Base::cellsSize();
    return;
  }
  // Erase the foreign particles.
  Base::eraseShadow(_localCellsBegin, _localCellsEnd);
  // Invalidate the exchange information.
  _isExchangePatternDefined = false;
  _localCellsBegin = 0;
  _localCellsEnd = Base::cellsSize();
  _receiving.clear();
  // Clear the sets of interior, exchanged, and shadow cells.
  interior.clear();
  exchanged.clear();
  shadow.clear();
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
printInfo(std::ostream& out) const
{
  Base::printInfo(out);
  out << "size = " << mpi::commSize(_comm) << '\n'
      << "rank = " << mpi::commRank(_comm) << '\n'
      << "delimiters = " << _delimiters[0];
  for (std::size_t i = 1; i != _delimiters.size(); ++i) {
    out << ", " << _delimiters[i];
  }
  out << '\n'
      << "_isExchangePatternDefined = " << _isExchangePatternDefined << '\n'
      << "_localCellsBegin = " << _localCellsBegin << '\n'
      << "_localCellsEnd = " << _localCellsEnd << '\n'
      << "receiving:\n";
  for (std::size_t i = 0; i != _receiving.size(); ++i) {
    out << "source = " << _receiving[i].source
        << ", position = " << _receiving[i].position
        << ", size = " << _receiving[i].size << '\n';
  }
  out << "_processCellLists:\n";
  for (std::size_t i = 0; i != _processCellLists.numArrays(); ++i) {
    if (! _processCellLists.empty(i)) {
      out << i << ": ";
      for (std::size_t j = 0; j != _processCellLists.size(i); ++j) {
        out << _processCellLists(i, j) << ' ';
      }
      out << '\n';
    }
  }
  out << "_startingImbalance = " << _startingImbalance << '\n'
      << "_partitionCount = " << _partitionCount << '\n'
      << "_reorderCount = " << _reorderCount << '\n';
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
printPerformanceInfo(std::ostream& out) const
{
  const std::size_t commRank = mpi::commRank(_comm);
  if (commRank == 0) {
    out << "Lower corner = " << Base::morton.lowerCorner() << '\n'
        << "Lengths = " << Base::morton.lengths() << '\n'
        << "Cell lengths = " << Base::morton.cellLengths() << '\n'
        << "Num levels = " << Base::morton.numLevels() << '\n'
        << "Cell extents = " << Base::morton.cellExtents() << '\n'
        << "Num cells capacity = " << Base::morton.maxCode() + 1 << "\n\n"
        << "Starting imbalance = " << _startingImbalance << '\n'
        << "Partition count = " << _partitionCount << '\n'
        << "Reorder count = " << _reorderCount << '\n'
        << "Repair count = " << _repairCount << '\n';
  }
  // Time totals.
  {
    const std::size_t Num = 5;
    const std::array<const char*, Num> names = {{
        "Reorder",
        "Partition",
        "Distribute",
        "ExPattern",
        "Exchange"
      }
    };
    const std::array<double, Num> measures = {
      {
        _timeReorder,
        _timePartition,
        _timeDistributeUnordered,
        _timeBuildExchangePattern,
        _timeExchangePost + _timeExchangeWait
      }
    };
    // Average.
    std::array<double, Num> averages;
    MPI_Reduce(&measures[0], &averages[0], measures.size(),
               MPI_DOUBLE, MPI_SUM, 0, _comm);
    averages /= double(mpi::commSize(_comm));
    // Minima.
    std::array<double, Num> minima;
    MPI_Reduce(&measures[0], &minima[0], measures.size(),
               MPI_DOUBLE, MPI_MIN, 0, _comm);
    // Maxima.
    std::array<double, Num> maxima;
    MPI_Reduce(&measures[0], &maxima[0], measures.size(),
               MPI_DOUBLE, MPI_MAX, 0, _comm);
    if (commRank == 0) {
      out << "\nTime totals:\n";
      // Column headers.
      for (std::size_t i = 0; i != names.size(); ++i) {
        out << ',' << names[i];
      }
      out << '\n';
      // Average.
      out << "Average";
      for (std::size_t i = 0; i != averages.size(); ++i) {
        out << ',' << averages[i];
      }
      out << '\n';
      // Minima.
      out << "Min";
      for (std::size_t i = 0; i != minima.size(); ++i) {
        out << ',' << minima[i];
      }
      out << '\n';
      // Maxima.
      out << "Max";
      for (std::size_t i = 0; i != maxima.size(); ++i) {
        out << ',' << maxima[i];
      }
      out << '\n';
    }
  }
  // Count totals.
  {
    const std::size_t Num = 9;
    const std::array<const char*, Num> names = {{
        "Local",
        "Shadow",
        "MinCellLength",
        "MaxCellLength",
        "Occupancy",
        "DistCount",
        "NeighborsSend",
        "NeighborsRecv",
        "ExCount"
      }
    };
    const std::size_t numLocalParticles =
      localParticlesEnd() - localParticlesBegin();
    const std::array<double, Num> measures = {{
        double(numLocalParticles),
        double(Base::particles.size() - numLocalParticles),
        ext::min(Base::morton.cellLengths()),
        ext::max(Base::morton.cellLengths()),
        double(numLocalParticles) /
        localCellsSize(),
        _numDistributeSent,
        _numNeighborsSend,
        _numNeighborsReceive,
        _numExchangeSent
      }
    };
    // Average.
    std::array<double, Num> averages;
    MPI_Reduce(&measures[0], &averages[0], measures.size(),
               MPI_DOUBLE, MPI_SUM, 0, _comm);
    averages /= double(mpi::commSize(_comm));
    // Minima.
    std::array<double, Num> minima;
    MPI_Reduce(&measures[0], &minima[0], measures.size(),
               MPI_DOUBLE, MPI_MIN, 0, _comm);
    // Maxima.
    std::array<double, Num> maxima;
    MPI_Reduce(&measures[0], &maxima[0], measures.size(),
               MPI_DOUBLE, MPI_MAX, 0, _comm);
    if (commRank == 0) {
      out << "\nCount totals:\n";
      // Column headers.
      for (std::size_t i = 0; i != names.size(); ++i) {
        out << ',' << names[i];
      }
      out << '\n';
      // Average.
      out << "Average";
      for (std::size_t i = 0; i != averages.size(); ++i) {
        out << ',' << averages[i];
      }
      out << '\n';
      // Minima.
      out << "Min";
      for (std::size_t i = 0; i != minima.size(); ++i) {
        out << ',' << minima[i];
      }
      out << '\n';
      // Maxima.
      out << "Max";
      for (std::size_t i = 0; i != maxima.size(); ++i) {
        out << ',' << maxima[i];
      }
      out << '\n';
    }
  }
  // Time per operation.
  {
    const std::size_t Num = 4;
    const std::array<const char*, Num> names = {{
        "Reorder",
        "Partition",
        "Distribute",
        "ExPattern"
      }
    };
    const std::array<double, Num> measures = {
      {
        _timeReorder / _reorderCount,
        _timePartition / _partitionCount,
        _timeDistributeUnordered / _partitionCount,
        _timeBuildExchangePattern / (1 + _reorderCount)
      }
    };
    // Average.
    std::array<double, Num> averages;
    MPI_Reduce(&measures[0], &averages[0], measures.size(),
               MPI_DOUBLE, MPI_SUM, 0, _comm);
    averages /= double(mpi::commSize(_comm));
    // Minima.
    std::array<double, Num> minima;
    MPI_Reduce(&measures[0], &minima[0], measures.size(),
               MPI_DOUBLE, MPI_MIN, 0, _comm);
    // Maxima.
    std::array<double, Num> maxima;
    MPI_Reduce(&measures[0], &maxima[0], measures.size(),
               MPI_DOUBLE, MPI_MAX, 0, _comm);
    if (commRank == 0) {
      out << "\nTime per operation:\n";
      // Column headers.
      for (std::size_t i = 0; i != names.size(); ++i) {
        out << ',' << names[i];
      }
      out << '\n';
      // Average.
      out << "Average";
      for (std::size_t i = 0; i != averages.size(); ++i) {
        out << ',' << averages[i];
      }
      out << '\n';
      // Minima.
      out << "Min";
      for (std::size_t i = 0; i != minima.size(); ++i) {
        out << ',' << minima[i];
      }
      out << '\n';
      // Maxima.
      out << "Max";
      for (std::size_t i = 0; i != maxima.size(); ++i) {
        out << ',' << maxima[i];
      }
      out << '\n';
    }
  }
  // Per step.
  {
    const std::size_t Num = 5;
    const std::array<const char*, Num> names = {{
        "ExPost",
        "ExWait",
        "NeighborsSend",
        "NeighborsRecv",
        "ExCount"
      }
    };
    const std::array<double, Num> measures = {
      {
        _timeExchangePost / (_repairCount + 1),
        _timeExchangeWait / (_repairCount + 1),
        _numNeighborsSend / (_repairCount + 1),
        _numNeighborsReceive / (_repairCount + 1),
        _numExchangeSent / (_repairCount + 1)
      }
    };
    // Average.
    std::array<double, Num> averages;
    MPI_Reduce(&measures[0], &averages[0], measures.size(),
               MPI_DOUBLE, MPI_SUM, 0, _comm);
    averages /= double(mpi::commSize(_comm));
    // Minima.
    std::array<double, Num> minima;
    MPI_Reduce(&measures[0], &minima[0], measures.size(),
               MPI_DOUBLE, MPI_MIN, 0, _comm);
    // Maxima.
    std::array<double, Num> maxima;
    MPI_Reduce(&measures[0], &maxima[0], measures.size(),
               MPI_DOUBLE, MPI_MAX, 0, _comm);
    if (commRank == 0) {
      out << "\nPer step:\n";
      // Column headers.
      for (std::size_t i = 0; i != names.size(); ++i) {
        out << ',' << names[i];
      }
      out << '\n';
      // Average.
      out << "Average";
      for (std::size_t i = 0; i != averages.size(); ++i) {
        out << ',' << averages[i];
      }
      out << '\n';
      // Minima.
      out << "Min";
      for (std::size_t i = 0; i != minima.size(); ++i) {
        out << ',' << minima[i];
      }
      out << '\n';
      // Maxima.
      out << "Max";
      for (std::size_t i = 0; i != maxima.size(); ++i) {
        out << ',' << maxima[i];
      }
      out << '\n';
    }
  }
  if (commRank == 0) {
    out << "\nMemory Usage:\n";
    Base::printMemoryUsageTable(out);
    out << "processCellLists,"
        << _processCellLists.memoryUsage() << ','
        << _processCellLists.memoryCapacity() << '\n'
        << "particleSendBuffers,"
        << _particleSendBuffers.memoryUsage() << ','
        << _particleSendBuffers.memoryCapacity() << '\n';
  }
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_printCellDataVtk(std::ostream& out, const std::size_t MaxImageExtent,
                  std::integral_constant<std::size_t, 3> /*Dimension*/) const
{
  // Make a table of the local cell data.
  std::vector<std::pair<Code, CellData> > table;
  _cellDataTable(&table);

  // For the maximum buffer size, take the minimum of a reasonable upper bound
  // on the number of cells and the number of voxels in the image.
  const std::size_t maxBufferSize =
    std::min(std::size_t(1.1 * table.size() * mpi::commSize(_comm)),
             MaxImageExtent * MaxImageExtent * MaxImageExtent);
  // Perform a binary reduction.
  std::size_t reducedLevels = _reduce(&table, maxBufferSize);

  if (mpi::commRank(_comm) != 0) {
    return;
  }

  // Make a Morton class at the appropriate level for the image.
  // Start with the current level for the table.
  Morton<Float, Dimension, Periodic> mortonImage = Base::morton;
  mortonImage.setLevels(mortonImage.numLevels() - reducedLevels);
  // Then see if we need to reduce the levels further to accomodate
  // the image.
  reducedLevels = 0;
  while (ext::max(mortonImage.cellExtents()) > MaxImageExtent) {
    mortonImage.setLevels(mortonImage.numLevels() - 1);
    ++reducedLevels;
  }
  if (reducedLevels != 0) {
    shift<Dimension>(&table, reducedLevels);
  }

  // Make a dense array for the image.
  CellData invalid = {std::numeric_limits<std::size_t>::max(),
                      std::numeric_limits<std::size_t>::max(),
                      std::numeric_limits<std::size_t>::max()
                     };
  typedef container::SimpleMultiArray<CellData, Dimension> MultiArray;
  typedef typename MultiArray::IndexList IndexList;
  MultiArray image(ext::convert_array<std::size_t>(mortonImage.cellExtents()),
                   invalid);

  // Set the voxels that are in the table.
  for (std::size_t i = 0; i != table.size(); ++i) {
    image(ext::convert_array<std::size_t>
          (mortonImage.coordinates(table[i].first))) = table[i].second;
  }

  // Fill in the gaps with data that has zero occupancy levels.
  CellData data = image[0];
  assert(data != invalid);
  IndexList coords;
  for (Code code = 0; code <= mortonImage.maxCode(); ++code) {
    coords = ext::convert_array<std::size_t>(mortonImage.coordinates(code));
    if (image(coords) == invalid) {
      image(coords) = data;
    }
    else {
      data = image(coords);
      data.occupancy = 0;
      data.sendCount = 0;
    }
  }

  // Write the VTK file.
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
      << "  <ImageData WholeExtent=\""
      << "0 " << image.extents()[0]
      << " 0 " << image.extents()[1]
      << " 0 " << image.extents()[2]
      << "\" Origin=\"" << mortonImage.lowerCorner()
      << "\" Spacing=\""
      << mortonImage.cellLengths()[0] << ' '
      << mortonImage.cellLengths()[1] << ' '
      << mortonImage.cellLengths()[2]
      << "\">\n"
      << "  <Piece Extent=\""
      << "0 " << image.extents()[0]
      << " 0 " << image.extents()[1]
      << " 0 " << image.extents()[2]
      << "\">\n"
      << "    <PointData>\n"
      << "    </PointData>\n"
      << "    <CellData>\n"
      << "      <DataArray type=\"UInt32\" Name=\"occupancy\" format=\"ascii\">\n";
  for (std::size_t i = 0; i != image.size(); ++i) {
    out << image[i].occupancy << ' ';
  }
  out << '\n'
      << "      </DataArray>\n"
      << "      <DataArray type=\"UInt32\" Name=\"sendCount\" format=\"ascii\">\n";
  for (std::size_t i = 0; i != image.size(); ++i) {
    out << image[i].sendCount << ' ';
  }
  out << '\n'
      << "      </DataArray>\n"
      << "      <DataArray type=\"UInt32\" Name=\"process\" format=\"ascii\">\n";
  for (std::size_t i = 0; i != image.size(); ++i) {
    out << image[i].process << ' ';
  }
  out << '\n'
      << "      </DataArray>\n"
      << "    </CellData>\n"
      << "  </Piece>\n"
      << "  </ImageData>\n"
      << "</VTKFile>\n";
}


template<typename _Traits>
inline
void
MortonOrderMpi<_Traits>::
_cellDataTable(std::vector<std::pair<Code, CellData> >* table) const
{
  const std::size_t commRank = mpi::commRank(_comm);
  table->clear();

  // Determine the sends counts for each non-empty cell.
  std::map<Code, std::size_t> sendCounts;
  // For each cell that will be sent to another process.
  for (std::size_t i = 0; i != _processCellLists.size(); ++i) {
    const Code code = Base::_cellCodes[_processCellLists[i]];
    if (sendCounts.count(code)) {
      sendCounts[code] += 1;
    }
    else {
      sendCounts[code] = 1;
    }
  }

  // Start with an empty cell that indicates the beginning of this process.
  const CellData empty = {0, 0, commRank};
  table->push_back(std::make_pair(_delimiters[commRank], empty));
  // Add the elements.
  for (std::size_t i = _localCellsBegin; i != _localCellsEnd; ++i) {
    CellData data = {Base::cellEnd(i) - Base::cellBegin(i), 0, commRank};
    const Code code = Base::_cellCodes[i];
    if (sendCounts.count(code)) {
      data.sendCount = sendCounts[code];
    }
    if (table->back().first == code) {
      table->back().second += data;
    }
    else {
      table->push_back(std::make_pair(code, data));
    }
  }
}

} // namespace particle
}

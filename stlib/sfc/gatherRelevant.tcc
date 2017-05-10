// -*- C++ -*-

#if !defined(__sfc_gatherRelevant_tcc__)
#error This file is an implementation detail of gatherRelevant.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Object, typename _Traits>
inline
_Object*
recordRelevant(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, void, true> const& objectCells,
  std::vector<typename _Traits::Code> const& relevantCodes,
  _Object* output)
{
  std::size_t c = 0;
  // For each of the relevant codes.
  for (std::size_t i = 0; i != relevantCodes.size(); ++i) {
    // Advance to the next relevant code.
    for (; c != objectCells.size() &&
           objectCells.code(c) < relevantCodes[i]; ++c) {
    }
    if (c == objectCells.size()) {
      break;
    }
    if (objectCells.code(c) == relevantCodes[i]) {
      // Record the objects.
      std::size_t const size = objectCells.delimiter(c + 1) -
        objectCells.delimiter(c);
      memcpy(output, &objects[objectCells.delimiter(c)],
             size * sizeof(_Object));
      output += size;
    }
  }
  return output;
}


template<typename _Object, typename _Traits, typename _ForwardIterator>
inline
std::vector<_Object>
recordRelevant(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, void, true> const& objectCells,
  _ForwardIterator relevantCodesBegin,
  _ForwardIterator const relevantCodesEnd)
{
  typedef typename _Traits::Code Code;

  std::vector<_Object> relevant;
  std::size_t c = 0;
  // For each of the relevant codes.
  for ( ; relevantCodesBegin != relevantCodesEnd; ++relevantCodesBegin) {
    Code const code = *relevantCodesBegin;
    // Advance to the next relevant code.
    for (; c != objectCells.size() && objectCells.code(c) < code; ++c) {
    }
    if (c == objectCells.size()) {
      break;
    }
    if (objectCells.code(c) == code) {
      // Record the objects.
      relevant.insert(relevant.end(), &objects[objectCells.delimiter(c)],
                      &objects[objectCells.delimiter(c + 1)]);
    }
  }
  return relevant;
}


// Determine the cells that are relevant for other processes. All indices are
// for the distributed cells.
inline
std::vector<std::size_t>
calculateRelevantForOthers
(std::size_t const numDistributedCells,
 std::vector<std::size_t> const& localCells,
 std::vector<std::size_t> const& localRelevantCells,
 MPI_Comm const comm)
{
#ifdef STLIB_DEBUG
  auto lessEqual = [](std::size_t a, std::size_t b){ return a <= b; };
  // The local cell indices must be strictly ascending.
  assert(std::is_sorted(localCells.begin(), localCells.end(), lessEqual));
  // The local cell indices must be in the allowed range.
  if (! localCells.empty()) {
    assert(localCells.back() < numDistributedCells);
  }
  // The local relevant cell indices must be strictly ascending.
  assert(std::is_sorted(localRelevantCells.begin(), localRelevantCells.end(),
                        lessEqual));
  // The local cell indices must be in the allowed range.
  if (! localRelevantCells.empty()) {
    assert(localRelevantCells.back() < numDistributedCells);
  }
#endif

  // We use the unsigned short type for accumulating relevancy counts.
  typedef unsigned short Count;
  // Verify that this integer type is adequate.
  if (std::size_t(mpi::commSize(comm)) > std::numeric_limits<Count>::max()) {
    throw std::runtime_error("The chosen number type is not sufficient for "
                             "this number of MPI processes.");
  }

  // Record the local counts.
  std::vector<Count> localCounts(numDistributedCells, 0);
  for (auto cell: localRelevantCells) {
    localCounts[cell] = 1;
  }

  // Perform an all-reduce to accumulate the counts.
  std::vector<Count> accumulatedCounts(localCounts.size());
  mpi::allReduce(localCounts, &accumulatedCounts, MPI_SUM, comm);

  // Subtract the local counts to get accumulated counts for the other
  // processes.
  for (auto cell: localRelevantCells) {
    --accumulatedCounts[cell];
  }

  // Record the local cells that are relevant for other processes.
  std::vector<std::size_t> relevantForOthers;
  for (auto cell: localCells) {
    if (accumulatedCounts[cell]) {
      relevantForOthers.push_back(cell);
    }
  }

  return relevantForOthers;
}


// Determine the cells that are relevant for other processes. The input cell
// indices are for the distributed cells. We return indices for the local cells.
template<typename _Traits, typename _Cell1, bool _StoreDel1, typename _Cell2,
         bool _StoreDel2>
inline
std::vector<std::size_t>
calculateRelevantForOthers
(stlib::sfc::AdaptiveCells<_Traits, _Cell1, _StoreDel1> const&
 distributedCells,
 stlib::sfc::AdaptiveCells<_Traits, _Cell2, _StoreDel2> const& localCells,
 std::vector<std::size_t> const& relevantDistributedCells,
 MPI_Comm const comm)
{
  typedef typename _Traits::Code Code;

  // Make a vector of the local cell codes.
  std::vector<Code> localCellCodes(localCells.size());
  for (std::size_t i = 0; i != localCells.size(); ++i) {
    localCellCodes[i] = localCells.code(i);
  }
  // Convert the codes to indices for the distributed cells.
  std::vector<std::size_t> localCellDistributedIndices =
    distributedCells.codesToCells(localCellCodes);

  // Calculate the relevant cells in terms of distributed cell indices.
  std::vector<std::size_t> relevantDistributedIndicesForOthers =
    calculateRelevantForOthers(distributedCells.size(),
                               localCellDistributedIndices,
                               relevantDistributedCells, comm);

  // Convert to local cell indices.
  return localCells.codesToCells(distributedCells.codes
                                 (relevantDistributedIndicesForOthers));
}


template<typename _Object, typename _Traits, typename _Cell>
inline
stlib::sfc::AdaptiveCells<_Traits, void, true>
buildLocalObjectCells(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, _Cell, true> const&
  distributedObjectCells)
{
  typedef typename _Traits::Code Code;
  typedef typename _Traits::BBox BBox;

  // Define the virtual grid.
  stlib::sfc::AdaptiveCells<_Traits, void, true>
    localObjectCells(distributedObjectCells.grid());

  // Calculate the object codes.
  std::vector<Code> objectCodes(objects.size());
  for (std::size_t i = 0; i != objectCodes.size(); ++i) {
    objectCodes[i] =
      localObjectCells.grid().code(centroid(geom::specificBBox<BBox>
                                            (objects[i])));
  }
#ifdef STLIB_DEBUG
  assert(std::is_sorted(objectCodes.begin(), objectCodes.end()));
#endif

  // Build the cells.
  localObjectCells.buildCells(distributedObjectCells.codesWithGuard(),
                              objectCodes, objects);
#ifdef STLIB_DEBUG
  localObjectCells.checkValidity();
#endif

  return localObjectCells;
}


template<typename _Object, typename _Traits, typename _Cell>
inline
std::vector<_Object>
gatherRelevantRing(
  std::vector<_Object>* objects,
  stlib::sfc::AdaptiveCells<_Traits, _Cell, true> const&
  distributedObjectCells,
  std::vector<std::size_t> const& relevantCells,
  MPI_Comm const comm)
{
  typedef typename _Traits::Code Code;
  using stlib::performance::start;
  using stlib::performance::stop;
  using stlib::performance::record;

  int const commSize = stlib::mpi::commSize(comm);
  int const commRank = stlib::mpi::commRank(comm);

  stlib::performance::Scope _("sfc::gatherRelevantRing()");
  record("Number of distributed cells", distributedObjectCells.size());
  record("Distributed cell storage", distributedObjectCells.storage());
  record("Number of local objects", objects->size());
  record("Number of relevant cells", relevantCells.size());
  start("Count the number of relevant objects");

  // Count the number of relevant objects for this process.
  std::size_t numRelevant = 0;
  for (std::size_t i = 0; i != relevantCells.size(); ++i) {
    // Accumulate the number of objects in the cell.
    numRelevant += distributedObjectCells.delimiter(relevantCells[i] + 1) -
      distributedObjectCells.delimiter(relevantCells[i]);
  }
  // Allocate memory for the relevant objects.
  std::vector<_Object> relevantObjects(numRelevant);
  // We will use this iterator for inserting relevant objects. This is more
  // efficient than using push_back().
  _Object* relevantObjectsIter = &relevantObjects[0];

  stop();
  start("Build object cells");

  // Build a cell data structure with delimiters for the local objects.
  stlib::sfc::AdaptiveCells<_Traits, void, true> localObjectCells =
    buildLocalObjectCells(*objects, distributedObjectCells);
  record("Number of object cells", localObjectCells.size());

  stop();
  start("Record the relevant objects.");

  // Record the relevant local objects.
  std::vector<Code> const relevantCodes =
    distributedObjectCells.codes(relevantCells);
  relevantObjectsIter =
    recordRelevant(*objects, localObjectCells, relevantCodes,
                   relevantObjectsIter);

  stop();
  record("Storage for local objects", objects->size() * sizeof(_Object));
  start("Calculate cells that are relevant for others");

  // Keep only the cells and objects that are relevant for other processes.
  {
    // Calculate the local cells that are relevant for other processes.
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(distributedObjectCells, localObjectCells,
                                 relevantCells, comm);
    // Crop to keep only the cells and objects that are relevant for other
    // processes.
    localObjectCells.crop(relevantForOthers, objects);
  }

  stop();
  record("Number of object cells that are relevant for others",
         localObjectCells.size());
  start("All gather the vector sizes.");

  // Serialize the cells.
  std::vector<unsigned char> sendCells;
  localObjectCells.serialize(&sendCells);
  // All gather the sizes of the vectors that we communicate, namely the
  // objects and the serialized cells.
  std::vector<std::pair<std::size_t, std::size_t> > const vectorSizes =
  stlib::mpi::allGather(std::pair<std::size_t, std::size_t>
                        (sendCells.size(), objects->size()), comm);

  stop();
  record("Storage for relevant objects",
         relevantObjects.size() * sizeof(_Object));
  start("Reserve memory.");

  // Reserve memory for the vectors that we use for communication. This
  // way we allocate memory only once.
  std::vector<unsigned char> recvCells;
  std::vector<_Object> recvObjects;
  {
    std::size_t cellMax = 0;
    std::size_t objectMax = 0;
    for (std::size_t i = 0; i != vectorSizes.size(); ++i) {
      if (vectorSizes[i].first > cellMax) {
        cellMax = vectorSizes[i].first;
      }
      if (vectorSizes[i].second > objectMax) {
        objectMax = vectorSizes[i].second;
      }
    }
    sendCells.reserve(cellMax);
    recvCells.reserve(cellMax);
    objects->reserve(objectMax);
    recvObjects.reserve(objectMax);
    record("Storage for send buffers", cellMax * sizeof(unsigned char) +
           objectMax * sizeof(_Object));
    record("Storage for receive buffers", cellMax * sizeof(unsigned char) +
           objectMax * sizeof(_Object));
  }

  stop();

  int const previousProcess = (commRank - 1 + commSize) % commSize;
  int const nextProcess = (commRank + 1) % commSize;
  enum {ObjectsTag, CellsTag};

  // Exchange objects commSize - 1 times.
  for (int i = 1; i != commSize; ++i) {
    // Note that blocking communication is just as fast on a shared-memory
    // system, but non-blocking communication is faster on a cluster.
    start("Initiate communication.");

    // The source of the chunk of the boundary that we will communicate.
    int const source = (commRank - i + commSize) % commSize;

    // Exchange the cell buffers.
    recvCells.resize(vectorSizes[source].first);
    MPI_Request recvCellsRequest =
      stlib::mpi::iRecv(&recvCells, previousProcess, CellsTag, comm);
    MPI_Request sendCellsRequest =
      stlib::mpi::iSend(sendCells, nextProcess, CellsTag, comm);

    // Exchange the objects.
    recvObjects.resize(vectorSizes[source].second);
    MPI_Request recvObjectsRequest =
      stlib::mpi::iRecv(&recvObjects, previousProcess, ObjectsTag, comm);
    MPI_Request sendObjectsRequest =
      stlib::mpi::iSend(*objects, nextProcess, ObjectsTag, comm);

    stop();
    start("Record the relevant objects.");

    // Record the relevant objects. Note that the first iteration is special
    // because we record the local relevant objects in the initialization.
    if (i != 1) {
      relevantObjectsIter =
        recordRelevant(*objects, localObjectCells, relevantCodes,
                       relevantObjectsIter);
    }

    stop();
    start("Wait and then unserialize the cells.");

    // Unserialize to get the cells.
    stlib::mpi::wait(&recvCellsRequest);
    localObjectCells.unserialize(recvCells);

    stop();
    start("Wait and then swap buffers.");

    // Swap because we will send the received cells in the next iteration.
    stlib::mpi::wait(&sendCellsRequest);
    sendCells.swap(recvCells);
    // Swap the object buffers.
    stlib::mpi::wait(&recvObjectsRequest);
    stlib::mpi::wait(&sendObjectsRequest);
    objects->swap(recvObjects);

    stop();
  }
  start("Record the relevant objects.");

  // Record the relevant objects in the final objects.
  relevantObjectsIter =
    recordRelevant(*objects, localObjectCells, relevantCodes,
                   relevantObjectsIter);
  assert(std::size_t(std::distance(&relevantObjects[0],
                                   relevantObjectsIter)) == 
         relevantObjects.size());

  stop();

  return relevantObjects;
}


template<typename _Object, typename _Traits, typename _Cell>
inline
std::vector<_Object>
gatherRelevantRing(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, _Cell, true> const&
  distributedObjectCells,
  std::vector<std::size_t> const& relevantCells,
  MPI_Comm const comm)
{
  std::vector<_Object> objectsCopy = objects;
  return gatherRelevantRing(&objectsCopy, distributedObjectCells, relevantCells,
                            comm);
}


template<typename _Object, typename _Traits, typename _Cell>
inline
std::vector<_Object>
gatherRelevantPointToPoint(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, _Cell, true> const&
  distributedObjectCells,
  std::vector<std::size_t> const& relevantCells,
  MPI_Comm const comm)
{
  typedef typename _Traits::Code Code;
  using stlib::performance::start;
  using stlib::performance::stop;
  using stlib::performance::record;

  int const commSize = stlib::mpi::commSize(comm);
  int const commRank = stlib::mpi::commRank(comm);

  stlib::performance::Scope _("sfc::gatherRelevantPointToPoint()");
  record("Number of distributed cells", distributedObjectCells.size());
  record("Distributed cell storage", distributedObjectCells.storage());
  record("Number of local objects", objects.size());
  record("Number of relevant cells", relevantCells.size());
  start("Count the number of relevant objects");

  // Count the number of relevant objects for this process.
  std::size_t numRelevant = 0;
  for (std::size_t i = 0; i != relevantCells.size(); ++i) {
    // Accumulate the number of objects in the cell.
    numRelevant += distributedObjectCells.delimiter(relevantCells[i] + 1) -
      distributedObjectCells.delimiter(relevantCells[i]);
  }
  // Allocate memory for the relevant objects.
  std::vector<_Object> relevantObjects(numRelevant);
  // We will use this iterator for inserting relevant objects. This is more
  // efficient than using push_back().
  _Object* relevantObjectsIter = &relevantObjects[0];

  stop();
  record("Storage for local objects", objects.size() * sizeof(_Object));
  record("Storage for relevant objects",
         relevantObjects.size() * sizeof(_Object));
  start("Build object cells");

  // Build a cell data structure with delimiters for the local objects.
  stlib::sfc::AdaptiveCells<_Traits, void, true> localObjectCells =
    buildLocalObjectCells(objects, distributedObjectCells);
  record("Number of object cells", localObjectCells.size());

  stop();
  start("Record the relevant objects.");

  // Record the relevant local objects.
  std::vector<Code> const relevantCodes =
    distributedObjectCells.codes(relevantCells);
  relevantObjectsIter =
    recordRelevant(objects, localObjectCells, relevantCodes,
                   relevantObjectsIter);

  stop();
  // This is commented out because it has little effect on performance.
  // (It typically hurts the performance by a small amount.) Note that enabling
  // This would require passing the objects by pointer.
#if 0
  start("Calculate cells that are relevant for others");

  // Keep only the cells and objects that are relevant for other processes.
  {
    // Calculate the local cells that are relevant for other processes.
    std::vector<std::size_t> relevantForOthers =
      calculateRelevantForOthers(distributedObjectCells, localObjectCells,
                                 relevantCells, comm);
    // Crop to keep only the cells and objects that are relevant for other
    // processes.
    localObjectCells.crop(relevantForOthers, objects);
  }

  stop();
  record("Number of object cells that are relevant for others",
         localObjectCells.size());
#endif
  start("All-gather the relevant cells");

  container::PackedArrayOfArrays<Code> processRelevantCodes =
    mpi::allGatherPacked(relevantCodes, comm);

  stop();
  start("Initiate sends for the objects");

  std::vector<std::size_t> areSendingToProcess(commSize, 0);
  std::vector<std::unique_ptr<std::vector<_Object> > > sendBuffers;
  std::vector<MPI_Request> sendRequests;
  // Loop over the ranks, starting with the next one. This reduces dependencies
  // in the communication pattern.
  for (int rank = (commRank + 1) % commSize; rank != commRank;
       rank = (rank + 1) % commSize) {
    // Record the objects for process rank.
    std::vector<_Object> buffer =
      recordRelevant(objects, localObjectCells,
                     processRelevantCodes.begin(rank),
                     processRelevantCodes.end(rank));
    if (! buffer.empty()) {
      // Record that we are sending a message to this process.
      areSendingToProcess[rank] = 1;
      // Create a send buffer that holds the objects to send.
      sendBuffers.push_back(std::unique_ptr<std::vector<_Object> >
                            (new std::vector<_Object>(std::move(buffer))));
      // Send the objects to process rank.
      sendRequests.push_back(mpi::iSend(*sendBuffers.back(), rank, 0, comm));
    }
  }

  stop();
  record("Number of sends", sendBuffers.size());
#ifdef STLIB_PERFORMANCE
  {
    std::size_t storage = 0;
    for (auto const& buffer: sendBuffers) {
      storage += buffer->size() * sizeof(_Object);
    }
    record("Storage for send buffers", storage);
  }
#endif
  start("Receive objects");

  // Calculate the number of messages that we will receive.
  std::vector<std::size_t> sendCounts;
  mpi::reduce(areSendingToProcess, &sendCounts, MPI_SUM, comm);
  std::size_t numMessages = mpi::scatter(sendCounts, comm);
  record("Number of receives", numMessages);
  // Receive each message.
  std::vector<_Object> buffer;
  while (numMessages--) {
    // Receive objects from any source. Note that the buffer will be resized
    // as needed.
    mpi::recv(&buffer, MPI_ANY_SOURCE, 0, comm);
    // Record the received objects.
    std::memcpy(relevantObjectsIter, &buffer[0],
                buffer.size() * sizeof(_Object));
    relevantObjectsIter += buffer.size();
  }

  stop();
  record("Storage for receive buffers", buffer.capacity() * sizeof(_Object));
  start("Wait for sends to complete");

  mpi::waitAll(&sendRequests);

  stop();

  assert(std::size_t(std::distance(&relevantObjects[0],
                                   relevantObjectsIter)) == 
         relevantObjects.size());

  return relevantObjects;
}


// Determine the processes to which we must send our cells. The process indices
// are enumerated in a ring that potentially starts with the following process.
// Note that the local process is not included.
inline
std::vector<int>
calculateRelevantProcesses
(std::size_t const numDistributedCells,
 std::vector<std::size_t> const& localCells,
 std::vector<std::size_t> const& localRelevantCells,
 MPI_Comm const comm)
{
  // Convert the relevant cell indices to a bit array.
  // All-gather the bit arrays.
  container::PackedArrayOfArrays<std::size_t> const relevantBitArrays =
    mpi::allGatherPacked(numerical::convertIndicesToBitArray<std::size_t>
                         (numDistributedCells, localRelevantCells), comm);

  // Convert the local cell indices to a bit array.
  std::vector<std::size_t> const localBitArray =
    numerical::convertIndicesToBitArray<std::size_t>(numDistributedCells,
                                                     localCells);

  // Process the bit arrays in a ring, starting with the next process.
  std::vector<int> relevantProcesses;
  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);
  for (int i = 1; i != commSize; ++i) {
    // Get the array for the n_th process.
    int const n = (commRank + i) % commSize;
    auto const b = relevantBitArrays.begin(n);
    // Check for common elements between local cells and relevant cells.
    for (std::size_t j = 0; j != localBitArray.size(); ++j) {
      if (localBitArray[j] & b[j]) {
        relevantProcesses.push_back(n);
        break;
      }
    }
  }
  return relevantProcesses;
}


// This would be used in the chain communication pattern. However, I have not
// implemented that because the point-to-point pattern is probably faster.
inline
std::size_t
countMessagesToReceive(std::vector<int> const& relevantProcesses,
                       MPI_Comm const comm)
{
  // We use the unsigned short type for accumulating relevancy counts.
  typedef unsigned short Count;
  int const commSize = mpi::commSize(comm);
  // Verify that this integer type is adequate.
  if (std::size_t(commSize) > std::numeric_limits<Count>::max()) {
    throw std::runtime_error("The chosen number type is not sufficient for "
                             "this number of MPI processes.");
  }

  // Record the local counts.
  std::vector<Count> localCounts(commSize, 0);
  for (auto i: relevantProcesses) {
    localCounts[i] = 1;
  }

  // Perform a reduction to accumulate the counts.
  std::vector<Count> accumulatedCounts;
  mpi::reduce(localCounts, &accumulatedCounts, MPI_SUM, comm);

  // Perform a scatter to distribute the values.
  return mpi::scatter(accumulatedCounts);
}


} // namespace sfc
} // namespace stlib

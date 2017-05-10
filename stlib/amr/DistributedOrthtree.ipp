// -*- C++ -*-

#if !defined(__amr_DistributedOrthtree_ipp__)
#error This file is an implementation detail of the class DistributedOrthtree.
#endif

namespace stlib
{
namespace amr
{

//----------------------------------------------------------------------------
// Partition.
//----------------------------------------------------------------------------

template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
partition()
{
  // Compute the partition delimiters.
  computePartitionDelimiters();

  // Determine the current delimiters.
  std::vector<SpatialIndex> currentDelimiters(_communicatorSize + 1);
  gatherDelimiters(&currentDelimiters);

  // Determine the processors from which we will receive nodes.
  std::vector<int> processesToReceiveFrom;
  computeProcessesToReceiveFrom(currentDelimiters,
                                std::back_inserter(processesToReceiveFrom));

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cout << "Number of processes from which we will receive nodes = "
              << processesToReceiveFrom.size() << "\n";
  }
#endif

  // Post receives for how many nodes we will receive from each process.
  std::vector<int>
  numberOfNodesToReceive(processesToReceiveFrom.size());
  std::vector<MPI::Request>
  numberOfNodesToReceiveRequests(processesToReceiveFrom.size());
  for (std::size_t i = 0; i != processesToReceiveFrom.size(); ++i) {
    numberOfNodesToReceiveRequests[i] =
      _communicator.Irecv(&numberOfNodesToReceive[i], 1, MPI::INT,
                          processesToReceiveFrom[i], NumberOfNodesMessage);
  }

  // Determine the processors to which we will send nodes.
  std::vector<int> processesToSendTo;
  computeProcessesToSendTo(currentDelimiters[_communicatorRank],
                           currentDelimiters[_communicatorRank + 1],
                           std::back_inserter(processesToSendTo));

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cout << "Number of processes to which we will send nodes = "
              << processesToSendTo.size() << "\n";
  }
#endif

  // Determine the nodes to send to each processor.
  std::vector<std::vector<iterator> > nodesToSend(processesToSendTo.size());
  // Store the number of nodes to send because the Isend needs a persistent
  // buffer.
  std::vector<int> numberOfNodesToSend(processesToSendTo.size());
  for (std::size_t i = 0; i != processesToSendTo.size(); ++i) {
    const std::size_t process = processesToSendTo[i];
    std::vector<iterator>& nodes = nodesToSend[i];
    computeNodesToSend(process, std::back_inserter(nodes));
    // Send the number of nodes that we are going to send.
    numberOfNodesToSend[i] = nodes.size();
    // We don't need to track the send request.
    _communicator.Isend(&numberOfNodesToSend[i], 1, MPI::INT,
                        process, NumberOfNodesMessage);
  }

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cout << "Number of nodes to send:\n";
    for (std::size_t i = 0; i != processesToSendTo.size(); ++i) {
      std::cout << processesToSendTo[i] << " "
                << numberOfNodesToSend[i] << "\n";
    }
  }
#endif

  // Wait for the number of nodes receives to complete.
  if (! processesToReceiveFrom.empty()) {
    MPI::Request::Waitall(numberOfNodesToReceiveRequests.size(),
                          &numberOfNodesToReceiveRequests[0]);
  }

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cout << "Number of nodes to receive:\n";
    for (std::size_t i = 0; i != processesToReceiveFrom.size(); ++i) {
      std::cout << processesToReceiveFrom[i] << " "
                << numberOfNodesToReceive[i] << "\n";
    }
  }
#endif

  //------------------------------------------------------------------------

  //
  // Post receives for the nodes we will receive from each relevant process.
  //
  std::vector<MessageInputStream> receiveStreams(processesToReceiveFrom.size());
  std::vector<MPI::Request>
  receiveNodesRequests(processesToReceiveFrom.size());
  for (std::size_t i = 0; i != processesToReceiveFrom.size(); ++i) {
    MessageInputStream& in = receiveStreams[i];
    // Make a buffer to hold the spatial indices and the patches.
    in.resize(numberOfNodesToReceive[i] *
              (SpatialIndex::getMessageStreamSize() +
               _helper->getMessageStreamSize()));
    receiveNodesRequests[i] =
      _communicator.Irecv(in.getData(), in.getSize(), MPI::BYTE,
                          processesToReceiveFrom[i], NodesMessage);
  }

  //
  // Post sends for the nodes.
  //
  std::vector<MessageOutputStream> sendStreams(processesToSendTo.size());
  std::vector<MPI::Request> sendNodesRequests(processesToSendTo.size());
  for (std::size_t i = 0; i != processesToSendTo.size(); ++i) {
    MessageOutputStream& out = sendStreams[i];
    const std::vector<iterator>& nodes = nodesToSend[i];
    out.reserve(numberOfNodesToSend[i] *
                (SpatialIndex::getMessageStreamSize() +
                 _helper->getMessageStreamSize()));
    for (typename std::vector<iterator>::const_iterator node = nodes.begin();
         node != nodes.end(); ++node) {
      out << (*node)->first;
      out << (*node)->second;
    }
    // Send the nodes.
    sendNodesRequests[i] =
      _communicator.Isend(out.getData(), out.getSize(), MPI::BYTE,
                          processesToSendTo[i], NodesMessage);
  }

  //
  // Erase the sent nodes.
  //
  for (std::size_t i = 0; i != processesToSendTo.size(); ++i) {
    // Wait for a node send to complete.
    const std::size_t index = MPI::Request::Waitany(sendNodesRequests.size(),
                              &sendNodesRequests[0]);
    std::vector<iterator>& nodes = nodesToSend[index];
    for (typename std::vector<iterator>::const_iterator node = nodes.begin();
         node != nodes.end(); ++node) {
      getOrthtree().erase(*node);
    }
  }
  // Free the memory for the nodes to send and the send buffers.
  // This is not required. The memory would be freed at the end of this
  // function. I'm just making room to insert nodes.
  {
    std::vector<std::vector<iterator> > tmp;
    nodesToSend.swap(tmp);
  }
  {
    std::vector<MessageOutputStream> tmp;
    sendStreams.swap(tmp);
  }

  //
  // Insert the received nodes.
  //
  for (std::size_t i = 0; i != processesToReceiveFrom.size(); ++i) {
    // Wait for a node receive to complete.
    const std::size_t index =
      MPI::Request::Waitany(receiveNodesRequests.size(),
                            &receiveNodesRequests[0]);
    SpatialIndex spatialIndex;
    MessageInputStream& in = receiveStreams[index];
    for (std::size_t j = 0; j != numberOfNodesToReceive[index]; ++j) {
      // Read the spatial index.
      in >> spatialIndex;
      // Insert and initialize the patch.
      const iterator node = _helper->insert(spatialIndex);
      // Read the patch.
      in >> node->second;
    }
  }

#if 0
  if (_communicatorRank == 0) {
    std::cout << "The orthree:\n" << getOrthtree();
  }
#endif
}


// Compute the partition delimiters.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
computePartitionDelimiters()
{
  // Get the node counts from all processors.
  std::vector<int> nodeCounts(_communicatorSize);
  const std::size_t localCount = getOrthtree().size();
  _communicator.Allgather(&localCount, 1, MPI::INT,
                          &nodeCounts[0], 1, MPI::INT);

  // Determine the nodes for this process. start and finish describe the
  // nodes that we currently have.
  std::size_t start = 0;
  for (std::size_t i = 0; i != _communicatorRank; ++i) {
    start += nodeCounts[i];
  }
  const std::size_t finish = start + nodeCounts[_communicatorRank];

  // Determine a partitioning of the nodes.
  // These delimiters include the beginning and end for each partition.
  std::vector<int> countDelimiters(_communicatorSize + 1);
  numerical::computePartitions
  (std::accumulate(nodeCounts.begin(), nodeCounts.end(), 0),
   _communicatorSize, countDelimiters.begin());

  // Determine which delimiters are in this process.
  const std::size_t indexBegin =
    std::lower_bound(countDelimiters.begin(),
                     countDelimiters.end(), start) - countDelimiters.begin();
  const std::size_t indexEnd =
    std::lower_bound(countDelimiters.begin(),
                     countDelimiters.end(), finish) - countDelimiters.begin();

  MessageOutputStream out;
  // If this process holds any of the delimiters.
  if (indexBegin != indexEnd) {
    out.reserve((indexEnd - indexBegin) *
                SpatialIndex::getMessageStreamSize());
  }
  for (std::size_t i = indexBegin; i != indexEnd; ++i) {
    const_iterator node = getOrthtree().begin();
#ifdef STLIB_DEBUG
    assert(start <= countDelimiters[i]);
    assert(countDelimiters[i] < finish);
    assert(countDelimiters[i] - start < getOrthtree().size());
#endif
    std::advance(node, countDelimiters[i] - start);
    // CONTINUE
#if 0
    if (_communicatorRank == 0) {
      std::cout << node->first << "\n";
    }
#endif
    out << node->first;
  }

  // Compute the receive counts and displacements for the receiving buffer.
  std::vector<int> receiveCounts(_communicatorSize);
  std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
  std::size_t partialSum = 0;
  std::size_t j = 0;
  for (std::size_t i = 0; i != _communicatorSize; ++i) {
    partialSum += nodeCounts[i];
    while (countDelimiters[j] < partialSum) {
      ++receiveCounts[i];
      ++j;
    }
  }
  const std::size_t numberDefined = std::accumulate(receiveCounts.begin(),
                                    receiveCounts.end(), 0);
  // CONTINUE
#if 0
  std::cout << "numberDefined = " << numberDefined << "\n";
#endif
  for (std::size_t i = 0; i != receiveCounts.size(); ++i) {
    receiveCounts[i] *= SpatialIndex::getMessageStreamSize();
  }
  std::vector<int> displacements(_communicatorSize);
  displacements[0] = 0;
  std::partial_sum(receiveCounts.begin(),
                   receiveCounts.begin() + _communicatorSize - 1,
                   displacements.begin() + 1);

  MessageInputStream in;

  //in.resize(_communicatorSize * SpatialIndex::getMessageStreamSize());
  in.resize(numberDefined * SpatialIndex::getMessageStreamSize());
  _communicator.Allgatherv(out.getData(), out.getSize(), MPI::BYTE,
                           in.getData(), &receiveCounts[0],
                           &displacements[0], MPI::BYTE);

  // The last delimiter is the invalid index. For the rest, initialize with
  // invalid in case there are more processes than nodes.
  {
    SpatialIndex invalid;
    invalid.invalidate();
    std::fill(_delimiters.begin(), _delimiters.end(), invalid);
  }
  // Set the delimiters that define the partition.
  for (std::size_t i = 0; i != numberDefined; ++i) {
    in >> _delimiters[i];
  }

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cerr << "SpatialIndex::getMessageStreamSize() = "
              << SpatialIndex::getMessageStreamSize()
              << ", in.getSize() = " << in.getSize() << "\n"
              << "out.getSize() = " << out.getSize() << "\n"
              << "receiveCounts = " << receiveCounts << "\n"
              << "displacements = " << displacements << "\n"
              << "delimiters =\n";
    for (std::size_t i = 0; i != _delimiters.size(); ++i) {
      std::cerr << _delimiters[i] << "\n";
    }
    std::cerr << "start = " << start << " finish = " << finish << "\n"
              << "indexBegin = " << indexBegin
              << " indexEnd = " << indexEnd << "\n"
              << "nodeCounts = " << nodeCounts << "\n"
              << "countDelimiters = " << countDelimiters << "\n";
  }
#endif
}

template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
template<typename _OutputIterator>
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
computeProcessesToReceiveFrom
(const std::vector<SpatialIndex>& currentDelimiters,
 _OutputIterator processes)
{
  const SpatialIndex lower = _delimiters[_communicatorRank];
  const SpatialIndex upper = _delimiters[_communicatorRank + 1];
  for (std::size_t p = 0; p != _communicatorSize; ++p) {
    // We won't receive from ourself.
    if (p == _communicatorRank) {
      continue;
    }
    // If the p_th process has any nodes.
    if (currentDelimiters[p] != currentDelimiters[p + 1]) {
      // If the range intersects our partition. Remember that the index
      // ranges are semi-open.
      if (currentDelimiters[p] < upper && currentDelimiters[p + 1] > lower) {
        *processes++ = p;
      }
    }
  }
}

// Determine the processors to which we will send nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
template<typename _OutputIterator>
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
computeProcessesToSendTo(const SpatialIndex& start, const SpatialIndex& finish,
                         _OutputIterator processes)
{
  // If we have no nodes, do nothing.
  if (getOrthtree().empty()) {
    return;
  }

  // Check each process.
  for (std::size_t p = 0; p != _communicatorSize; ++p) {
    // We won't send to ourself.
    if (p == _communicatorRank) {
      continue;
    }
    // If our range intersects their partition.
    if (start < _delimiters[p + 1] && finish > _delimiters[p]) {
      *processes++ = p;
    }
  }
}

// Determine the nodes to send to the specified process.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
template<typename _OutputIterator>
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
computeNodesToSend(const std::size_t process, _OutputIterator nodes)
{
  const SpatialIndex& lower = _delimiters[process];
  const SpatialIndex& upper = _delimiters[process + 1];
  for (iterator node = getOrthtree().begin(); node != getOrthtree().end();
       ++node) {
    if (lower <= node->first && node->first < upper) {
      *nodes++ = node;
    }
  }
}

// Gather the spatial index delimiters.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
gatherDelimiters(std::vector<SpatialIndex>* currentDelimiters)
{
  assert(currentDelimiters->size() == _communicatorSize + 1);

  // Record the first spatial index for this processor.
  SpatialIndex invalid;
  invalid.invalidate();
  MessageOutputStream out(SpatialIndex::getMessageStreamSize());
  if (! getOrthtree().empty()) {
    out << getOrthtree().begin()->first;
  }
  else {
    // This is the flag that this process has no nodes.
    out << invalid;
  }
  // Gather the spatial indices from all processes.
  MessageInputStream in;
  in.resize(_communicatorSize * SpatialIndex::getMessageStreamSize());
  _communicator.Allgather(out.getData(), out.getSize(), MPI::BYTE,
                          in.getData(), out.getSize(), MPI::BYTE);
  // Set the delimiters.
  for (std::size_t i = 0; i != _communicatorSize; ++i) {
    in >> (*currentDelimiters)[i];
  }
  (*currentDelimiters)[_communicatorSize].invalidate();

  // Some processes may have no nodes. Apply a fix for this case.
  for (std::size_t i = _communicatorSize - 1; i >= 0; --i) {
    if (!(*currentDelimiters)[i].isValid()) {
      (*currentDelimiters)[i] = (*currentDelimiters)[i + 1];
    }
  }

  // CONTINUE
#if 0
  if (_communicatorRank == 0) {
    std::cout << "current delimiters:\n";
    for (std::size_t i = 0; i != currentDelimiters->size(); ++i) {
      std::cout << (*currentDelimiters)[i] << "\n";
    }
  }
#endif
}

//----------------------------------------------------------------------------
// Exchange.
//----------------------------------------------------------------------------

// Set up the exchange of adjacent nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacentSetUp()
{
  exchangeAdjacentDetermineNodesToSend();

  // The processes to which we will send nodes are the same processes from
  // which we will receive nodes. Determine how many nodes we will receive
  // from each process.
  exchangeAdjacentDetermineHowManyNodesToReceive();

  // Exchange the nodes keys. Insert ghost nodes for the nodes we will be
  // receiving.
  exchangeAdjacentSpatialIndicesAndInsertGhostNodes();
}


// Exchange the adjacent nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacent()
{
  // The number of processes with which we will exchange nodes.
  const std::size_t numberOfMessages = _nodesToSend.size();

  // Post receives for the patches.
  std::vector<MessageInputStream> receiveStreams(numberOfMessages);
  std::vector<MPI::Request> receiveRequests(numberOfMessages);
  {
    typename std::map<int, std::vector<iterator> >::const_iterator
    mapIterator = _nodesToReceive.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      MessageInputStream& in = receiveStreams[i];
      in.resize(mapIterator->second.size() *
                _helper->getMessageStreamSize());
      receiveRequests[i] =
        _communicator.Irecv(in.getData(), in.getSize(), MPI::BYTE,
                            mapIterator->first, PatchMessage);
    }
  }

  // Send the patches.
  std::vector<MessageOutputStream> sendStreams(numberOfMessages);
  std::vector<MPI::Request> sendRequests(numberOfMessages);
  {
    typename std::map<int, std::vector<const_iterator> >::const_iterator
    mapIterator = _nodesToSend.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      // The nodes for the patches that we will send.
      const std::vector<const_iterator>& nodes = mapIterator->second;
      // Pack the patches into the buffer.
      MessageOutputStream& out = sendStreams[i];
      out.reserve(nodes.size() * _helper->getMessageStreamSize());
      for (typename std::vector<const_iterator>::const_iterator node =
             nodes.begin(); node != nodes.end(); ++node) {
        out << (*node)->second;
      }
      // Send the buffer.
      sendRequests[i] =
        _communicator.Isend(out.getData(), out.getSize(), MPI::BYTE,
                            mapIterator->first, PatchMessage);
    }
  }

  //
  // Set the ghost nodes.
  //
  MPI::Status status;
  for (std::size_t i = 0; i != numberOfMessages; ++i) {
    // Wait for a receive to complete.
    const std::size_t index =
      MPI::Request::Waitany(receiveRequests.size(), &receiveRequests[0],
                            status);
    // The receiving buffer.
    MessageInputStream& in = receiveStreams[index];
    // The iterators to the ghost nodes from a process.
    std::vector<iterator>& nodes = _nodesToReceive[status.Get_source()];
    assert(nodes.size() * _helper->getMessageStreamSize() == in.getSize());

    for (std::size_t j = 0; j != nodes.size(); ++j) {
      // Read the patch.
      in >> nodes[j]->second;
      // Mark the patch as a ghost.
      nodes[j]->second.setGhost();
    }
  }

  // Wait for the sends to complete.
  MPI::Request::Waitall(sendRequests.size(), &sendRequests[0]);
}


// Tear down the data structure for the exchange of adjacent nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacentTearDown()
{
  //
  // Remove the ghost nodes.
  //
  // For each process from which we receive ghost nodes.
  for (typename std::map<int, std::vector<iterator> >::const_iterator
       i = _nodesToReceive.begin(); i != _nodesToReceive.end(); ++i) {
    const std::vector<iterator>& nodes = i->second;
    // For the ghost nodes we receive from that process.
    for (typename std::vector<iterator>::const_iterator node = nodes.begin();
         node != nodes.end(); ++node) {
      getOrthtree().erase(*node);
    }
  }

  //
  // Clear the exchange data structures.
  //
  _nodesToSend.clear();
  _nodesToReceive.clear();
}


// Determine the nodes to send in the exchange of adjacent nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacentDetermineNodesToSend()
{
  const SpatialIndex lower = _delimiters[_communicatorRank];
  const SpatialIndex upper = _delimiters[_communicatorRank + 1];

  // Check that they did not forget to call exchangeAdjacentTearDown().
  assert(_nodesToSend.empty());

  // For each node.
  std::vector<SpatialIndex> neighbors;
  for (const_iterator node = getOrthtree().begin();
       node != getOrthtree().end(); ++node) {
    //
    // Get the adjacent neighbors.
    //
    neighbors.clear();
    // If this node is not at the highest level.
    if (hasChildren(node->first)) {
      // Get the adjacent neighbors at the next highest level.
      getAdjacentNeighborsHigherLevel(node->first,
                                      std::back_inserter(neighbors));
    }
    else {
      // Get the adjacent neighbors at the same level.
      getAdjacentNeighbors(node->first, std::back_inserter(neighbors));
    }

    // For each adjacent neighbor.
    for (typename std::vector<SpatialIndex>::const_iterator
         neighbor = neighbors.begin(); neighbor != neighbors.end();
         ++ neighbor) {
      // If the neighbor is not in this process.
      if (!(lower <= *neighbor && *neighbor < upper)) {
        // Add this node to the send list for the appropriate process.
        // Note that the process may get multiple instances of this node.
        // We correct that later.
        // CONTINUE: Using getProcess is costly: O(log(P)). I could store and
        // then sort to avoid the binary searches.
        const std::size_t process = getProcess(*neighbor);
#ifdef STLIB_DEBUG
        assert(process != _communicatorRank);
#endif
        _nodesToSend[process].push_back(node);
      }
    }
  }

  // The send lists may have multiple copies of iterators. Remove the
  // duplicates.
  typename OrthtreeType::CompareIterator compare;
  std::vector<const_iterator> unique;
  for (typename std::map<int, std::vector<const_iterator> >::iterator
       i = _nodesToSend.begin(); i != _nodesToSend.end(); ++i) {
    std::vector<const_iterator>& nodeVector = i->second;
    // Remove the duplicates.
    std::sort(nodeVector.begin(), nodeVector.end(), compare);
    unique.clear();
    std::unique_copy(nodeVector.begin(), nodeVector.end(),
                     std::back_inserter(unique));
    nodeVector.swap(unique);
  }
}


// Determine the nodes to send in the exchange of adjacent nodes.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacentDetermineHowManyNodesToReceive()
{
  // The number of processes with which we will exchange nodes.
  const std::size_t numberOfMessages = _nodesToSend.size();

  // Store the node send counts in a vector so we can send them by address.
  std::vector<int> processes(numberOfMessages);
  std::vector<int> nodeSendCounts(numberOfMessages);
  {
    typename std::map<int, std::vector<const_iterator> >::const_iterator
    mapIterator = _nodesToSend.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      processes[i] = mapIterator->first;
      nodeSendCounts[i] = mapIterator->second.size();
    }
  }

  // Post receives for the node counts.
  std::vector<int> nodeReceiveCounts(numberOfMessages);
  std::vector<MPI::Request> receiveRequests(numberOfMessages);
  for (std::size_t i = 0; i != numberOfMessages; ++i) {
    receiveRequests[i] =
      _communicator.Irecv(&nodeReceiveCounts[i], 1, MPI::INT,
                          processes[i], NumberOfNodesMessage);
  }

  // Send the node counts.
  std::vector<MPI::Request> sendRequests(numberOfMessages);
  for (std::size_t i = 0; i != numberOfMessages; ++i) {
    sendRequests[i] =
      _communicator.Isend(&nodeSendCounts[i], 1, MPI::INT,
                          processes[i], NumberOfNodesMessage);
  }

  // Wait for the receives to complete.
  MPI::Request::Waitall(receiveRequests.size(), &receiveRequests[0]);

  // Check that they did not forget to call exchangeAdjacentTearDown().
  assert(_nodesToReceive.empty());
  // Set up the data structure for receiving the nodes.
  {
    typename std::map<int, std::vector<const_iterator> >::const_iterator
    mapIterator = _nodesToSend.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      _nodesToReceive[mapIterator->first].resize(nodeReceiveCounts[i]);
    }
  }

  // Wait for the sends to complete.
  MPI::Request::Waitall(sendRequests.size(), &sendRequests[0]);
}


// Exchange the nodes keys. Insert ghost nodes for the nodes we will be
// receiving.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
void
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
exchangeAdjacentSpatialIndicesAndInsertGhostNodes()
{
  // The number of processes with which we will exchange nodes.
  const std::size_t numberOfMessages = _nodesToSend.size();

  // Post receives for the spatial indices.
  std::vector<MessageInputStream> receiveStreams(numberOfMessages);
  std::vector<MPI::Request> receiveRequests(numberOfMessages);
  {
    typename std::map<int, std::vector<iterator> >::const_iterator
    mapIterator = _nodesToReceive.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      MessageInputStream& in = receiveStreams[i];
      in.resize(mapIterator->second.size() *
                SpatialIndex::getMessageStreamSize());
      receiveRequests[i] =
        _communicator.Irecv(in.getData(), in.getSize(), MPI::BYTE,
                            mapIterator->first, SpatialIndexMessage);
    }
  }

  // Send the spatial indices.
  std::vector<MessageOutputStream> sendStreams(numberOfMessages);
  std::vector<MPI::Request> sendRequests(numberOfMessages);
  {
    typename std::map<int, std::vector<const_iterator> >::const_iterator
    mapIterator = _nodesToSend.begin();
    for (std::size_t i = 0; i != numberOfMessages; ++i, ++mapIterator) {
      // The nodes that we will send.
      const std::vector<const_iterator>& nodes = mapIterator->second;
      // Pack the spatial indices into the buffer.
      MessageOutputStream& out = sendStreams[i];
      out.reserve(nodes.size() * SpatialIndex::getMessageStreamSize());
      for (typename std::vector<const_iterator>::const_iterator node =
             nodes.begin(); node != nodes.end(); ++node) {
        out << (*node)->first;
      }
      // Send the buffer.
      sendRequests[i] =
        _communicator.Isend(out.getData(), out.getSize(), MPI::BYTE,
                            mapIterator->first, SpatialIndexMessage);
    }
  }

  //
  // Insert the ghost nodes.
  //
  MPI::Status status;
  SpatialIndex spatialIndex;
  for (std::size_t i = 0; i != numberOfMessages; ++i) {
    // Wait for a receive to complete.
    const std::size_t index =
      MPI::Request::Waitany(receiveRequests.size(),
                            &receiveRequests[0], status);
    // The receiving buffer.
    MessageInputStream& in = receiveStreams[index];
    // The iterators to the ghost nodes from a process.
    std::vector<iterator>& nodes = _nodesToReceive[status.Get_source()];
    assert(nodes.size() * SpatialIndex::getMessageStreamSize() == in.getSize());

    for (std::size_t j = 0; j != nodes.size(); ++j) {
      // Read the spatial index.
      in >> spatialIndex;
      // Insert and initialize the patch.
      nodes[j] = _helper->insert(spatialIndex);
      // Mark the patch as a ghost.
      nodes[j]->second.setGhost();
    }
  }

  // Wait for the sends to complete.
  MPI::Request::Waitall(sendRequests.size(), &sendRequests[0]);
}

//----------------------------------------------------------------------------
// Exchange.
//----------------------------------------------------------------------------

// Perform refinement to balance the tree.
template < class _Patch, class _Traits,
           template<class _Patch, class _Traits> class _PatchHelper >
inline
std::size_t
DistributedOrthtree<_Patch, _Traits, _PatchHelper>::
balance()
{
  std::size_t total = 0, count, countSum;
  // Loop until no more refinement operations are required.
  do {
    // Set up the exchange to get the ghost nodes.
    exchangeAdjacentSetUp();
    // Perform local balancing.
    count = getOrthtree().balance();
    // Clean up.
    exchangeAdjacentTearDown();
    // Get the sum of refinements for this step.
    _communicator.Allreduce(&count, &countSum, 1, MPI::INT, MPI::SUM);
    total += countSum;
  }
  while (countSum != 0);
  return total;
}

} // namespace amr
}

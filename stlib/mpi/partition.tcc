// -*- C++ -*-

#if !defined(__mpi_partition_tcc__)
#error This file is an implementation detail of partition.
#endif

namespace stlib
{
namespace mpi
{


template<typename _T>
inline
bool
partitionOrdered(std::vector<_T>* objects, MPI_Comm const comm,
                 double const maxImbalance)
{
  // Gather the number of objects.
  std::vector<std::size_t> const sizes = mpi::allGather(objects->size(), comm);
  std::size_t const numProcs = sizes.size();
  std::size_t const rank = commRank(comm);
  // Check the load imbalance. If it is acceptable, do nothing.
  std::size_t const numObjects = ext::sum(sizes);
  if (numObjects == 0 ||
      ext::max(sizes) / (double(numObjects) / numProcs) <= 1 + maxImbalance) {
    return false;
  }

  // The delimiters for the initial partitioning of the objects.
  std::vector<std::size_t> initialDelimiters(numProcs + 1);
  initialDelimiters[0] = 0;
  for (std::size_t i = 0; i != sizes.size(); ++i) {
    initialDelimiters[i + 1] = initialDelimiters[i] + sizes[i];
  }

  // Calculate a fair partitioning of the distributed objects.
  std::vector<std::size_t> newDelimiters;
  numerical::computePartitions(numObjects, numProcs,
                               std::back_inserter(newDelimiters));

  // Determine the counts for the objects that we will send.
  std::vector<int> sendCounts(numProcs, 0);
  {
    std::size_t begin = initialDelimiters[rank];
    std::size_t const end = initialDelimiters[rank + 1];
    // Find the first process whose new range overlaps ours.
    std::size_t i =
      std::distance(newDelimiters.begin(),
                    std::upper_bound(newDelimiters.begin(), newDelimiters.end(),
                                     begin)) - 1;
    // Loop until we have determined where to send objects.
    for (; begin != end; ++i) {
      sendCounts[i] = std::min(end - begin, newDelimiters[i + 1] - begin);
      begin += sendCounts[i];
    }
  }

  // Determine the counts for the objects that we will receive.
  std::vector<int> recvCounts(numProcs, 0);
  {
    std::size_t begin = newDelimiters[rank];
    std::size_t const end = newDelimiters[rank + 1];
    // Find the first process whose initial range overlaps ours.
    std::size_t i =
      std::distance(initialDelimiters.begin(),
                    std::upper_bound(initialDelimiters.begin(),
                                     initialDelimiters.end(),
                                     begin)) - 1;
    // Loop until we have determined from where to receive objects.
    for (; begin != end; ++i) {
      recvCounts[i] = std::min(end - begin, initialDelimiters[i + 1] - begin);
      begin += recvCounts[i];
    }
  }

  // Exchange the objects.
  std::vector<_T> newObjects(newDelimiters[rank + 1] - newDelimiters[rank]);
  mpi::allToAll(&(*objects)[0], &sendCounts[0], &newObjects[0], &recvCounts[0],
                comm);
  objects->swap(newObjects);

  return true;
}


template<typename _T>
inline
bool
partitionExcess(std::vector<_T>* objects, MPI_Comm const comm,
                double const maxImbalance)
{
  // Gather the number of objects.
  std::vector<std::size_t> const sizes = mpi::allGather(objects->size(), comm);
  std::size_t const numProcs = sizes.size();
  std::size_t const rank = commRank(comm);
  // Check the load imbalance. If it is acceptable, do nothing.
  std::size_t const numObjects = ext::sum(sizes);
  if (numObjects == 0 ||
      ext::max(sizes) / (double(numObjects) / numProcs) <= 1 + maxImbalance) {
    return false;
  }

  // Calculate a fair partitioning of the distributed objects.
  std::vector<std::size_t> newDelimiters;
  numerical::computePartitions(numObjects, numProcs,
                               std::back_inserter(newDelimiters));

  // Calculate the excess and deficits.
  std::vector<std::size_t> excess(numProcs, 0);
  std::vector<std::size_t> deficits(numProcs, 0);
  for (std::size_t i = 0; i != numProcs; ++i) {
    if (sizes[i] > newDelimiters[i + 1] - newDelimiters[i]) {
      excess[i] = sizes[i] - (newDelimiters[i + 1] - newDelimiters[i]);
    }
    else {
      deficits[i] = (newDelimiters[i + 1] - newDelimiters[i]) - sizes[i];
    }
  }

  // The delimiters for objects to send and receive.
  std::vector<std::size_t> sendDelimiters(numProcs + 1);
  sendDelimiters[0] = 0;
  for (std::size_t i = 0; i != numProcs; ++i) {
    sendDelimiters[i + 1] = sendDelimiters[i] + excess[i];
  }
  std::vector<std::size_t> receiveDelimiters(numProcs + 1);
  receiveDelimiters[0] = 0;
  for (std::size_t i = 0; i != numProcs; ++i) {
    receiveDelimiters[i + 1] = receiveDelimiters[i] + deficits[i];
  }

  // Determine the counts for the objects that we will send.
  std::vector<int> sendCounts(numProcs, 0);
  {
    std::size_t begin = sendDelimiters[rank];
    std::size_t const end = sendDelimiters[rank + 1];
    // Find the first process whose new range overlaps ours.
    std::size_t i =
      std::distance(receiveDelimiters.begin(),
                    std::upper_bound(receiveDelimiters.begin(),
                                     receiveDelimiters.end(), begin)) - 1;
    // Loop until we have determined where to send objects.
    for (; begin != end; ++i) {
      sendCounts[i] = std::min(end - begin, receiveDelimiters[i + 1] - begin);
      begin += sendCounts[i];
    }
  }
#ifdef DEBUG_STLIB
  assert(ext::sum(sendCounts) == excess[rank];
#endif

#if 0
  // CONTINUE REMOVE
  if (rank == 0) {
    std::cerr << "sizes\n" << sizes << '\n'
              << "newDelimiters\n" << newDelimiters << '\n'
              << "excess\n" << excess << '\n'
              << "deficits\n" << deficits << '\n'
              << "sendDelimiters\n" << sendDelimiters << '\n'
              << "receiveDelimiters\n" << receiveDelimiters << '\n'
              << "sendCounts\n" << sendCounts << '\n';
  }
  MPI_Barrier(comm);
#endif

  // Determine the counts for the objects that we will receive.
  std::vector<int> recvCounts(numProcs, 0);
  {
    std::size_t begin = receiveDelimiters[rank];
    std::size_t const end = receiveDelimiters[rank + 1];
    // Find the first process whose initial range overlaps ours.
    std::size_t i =
      std::distance(sendDelimiters.begin(),
                    std::upper_bound(sendDelimiters.begin(),
                                     sendDelimiters.end(), begin)) - 1;
    // Loop until we have determined from where to receive objects.
    for (; begin != end; ++i) {
      recvCounts[i] = std::min(end - begin, sendDelimiters[i + 1] - begin);
      begin += recvCounts[i];
    }
  }

  // Exchange the objects.
  std::vector<_T> newObjects(newDelimiters[rank + 1] - newDelimiters[rank]);
  std::size_t const keep = objects->size() - excess[rank];
  std::copy(objects->begin(), objects->begin() + keep, newObjects.begin());
  mpi::allToAll(&(*objects)[keep], &sendCounts[0], &newObjects[keep],
                &recvCounts[0], comm);
  objects->swap(newObjects);

  return true;
}


} // namespace mpi
}

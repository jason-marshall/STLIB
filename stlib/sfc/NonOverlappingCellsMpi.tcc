// -*- C++ -*-

#if !defined(__sfc_NonOverlappingCellsMpi_tcc__)
#error This file is an implementation detail of NonOverlappingCellsMpi.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
inline
void
send(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order> const& input,
     int const dest, int const tag, MPI_Comm const comm)
{
  std::vector<unsigned char> buffer;
  input.serialize(&buffer);
  mpi::send(buffer, dest, tag, comm);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
inline
void
recv(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* output,
     int const source, int const tag, MPI_Comm const comm)
{
  // Receive the buffer of serialized data.
  std::vector<unsigned char> buffer;
  mpi::recv(&buffer, source, tag, comm);
  // Unserialize to get the cells.
  output->unserialize(buffer);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
inline
void
bcast(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* cells,
      MPI_Comm const comm, int const root)
{
  int const commRank = mpi::commRank(comm);

  // Serialize the cells on the root process.
  std::vector<unsigned char> buffer;
  if (commRank == root) {
    cells->serialize(&buffer);
  }
  // Broadcast the buffer.
  mpi::bcast(&buffer, comm, root);
  // Unserialize in the non-root processes.
  if (commRank != root) {
    cells->unserialize(buffer);
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
inline
void
reduceCompatible
(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order> const& input,
 NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* output,
 MPI_Comm const comm)
{
  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);

  // A power of two that is at least the number of processes.
  int n = 1;
  while (n < commSize) {
    n *= 2;
  }

  int tag = 0;
  NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>
    received(input.grid());
  *output = input;
  for (; n > 1; n /= 2) {
    // If this process is a receiver.
    if (commRank < n / 2) {
      int const sender = commRank + n / 2;
      // If there is a sender.
      if (sender < commSize) {
        // Receive a group of cells.
        recv(&received, sender, tag, comm);
        // Merge into the output.
        *output += received;
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      int const receiver = commRank - n / 2;
      // Send the cells.
      send(*output, receiver, tag, comm);
      output->clear();
    }
    ++tag;
  }
}


template<typename _Traits, typename _Cell, template<typename> class _Order,
         typename _Object>
inline
void
distributeNoSort
(NonOverlappingCells<_Traits, _Cell, true, _Order> const& localCells,
 std::vector<_Object>* objects,
 Partition<_Traits> const& codePartition, MPI_Comm const comm)
{
  // MPI bookkeeping.
  int const commSize = mpi::commSize(comm);

  // Calculate delimiters that determine the cells that we will send to
  // each process.
  std::vector<std::size_t> cellsToSend(commSize + 1);
  cellsToSend[0] = 0;
  std::size_t i = 0;
  for (std::size_t process = 1; process != cellsToSend.size() - 1; ++process) {
    while (localCells.code(i) < codePartition.delimiters[process]) {
      ++i;
    }
    cellsToSend[process] = i;
  }
  cellsToSend.back() = localCells.size();
#ifdef DEBUG_STLIB
  for (std::size_t process = 0; process != cellsToSend.size() - 1; ++process) {
    for (std::size_t i = cellsToSend[process]; i != cellsToSend[process + 1];
         ++i) {
      assert(codePartition.delimiters[process] <= localCells.code(i) &&
             localCells.code(i) < codePartition.delimiters[process + 1]);
    }
  }
#endif

  // Exchange to get the local objects for this process.
  {
    // The number of objects that we will send to each process.
    std::vector<std::size_t> sendCounts(commSize);
    for (std::size_t i = 0; i != sendCounts.size(); ++i) {
      sendCounts[i] = 0;
      // Loop over the cells that we will send to process i.
      for (std::size_t j = cellsToSend[i]; j != cellsToSend[i + 1]; ++j) {
        // Add the number of objects in the j_th cell.
        sendCounts[i] += localCells.delimiter(j + 1) - localCells.delimiter(j);
      }
    }
    std::vector<_Object> recvBuf;
    // Perform the all-to-all communication.
    mpi::allToAll(*objects, sendCounts, &recvBuf, comm);
    objects->swap(recvBuf);
  }
}


} // namespace sfc
} // namespace stlib

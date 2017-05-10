// -*- C++ -*-

#if !defined(__mpi_allToAll_tcc__)
#error This file is an implementation detail of allToAll.
#endif

namespace stlib
{
namespace mpi
{


template<typename _T>
inline
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         container::PackedArrayOfArrays<_T>* receive,
         MPI_Comm const comm)
{
  // Check that the number of arrays matches the number of processes.
  std::size_t const numProcesses = send.numArrays();
  assert(numProcesses == std::size_t(mpi::commSize(comm)));

  // Record the number of objects that we will send to each process.
  std::vector<int> sendCounts(numProcesses);
  for (std::size_t i = 0; i != sendCounts.size(); ++i) {
    sendCounts[i] = send.size(i);
  }
  // Record the send displacements.
  std::vector<int> sendDisplacements(numProcesses);
  for (std::size_t i = 0; i != sendDisplacements.size(); ++i) {
    sendDisplacements[i] = send.begin(i) - send.begin();
  }

  // Determine the number of objects that we will receive from each process.
  std::vector<int> receiveCounts(numProcesses);
  allToAll(sendCounts, &receiveCounts, comm);
  // Allocate memory for the objects we will receive.
  receive->rebuild(receiveCounts.begin(), receiveCounts.end());
  // Record the receive displacements.
  std::vector<int> receiveDisplacements(numProcesses);
  for (std::size_t i = 0; i != receiveDisplacements.size(); ++i) {
    receiveDisplacements[i] = receive->begin(i) - receive->begin();
  }

  // Perform the all-to-all communication.
  Data<_T> const data;
  allToAll(send.data(), &sendCounts[0], &sendDisplacements[0], data.type(),
           receive->data(), &receiveCounts[0], &receiveDisplacements[0],
           data.type(), comm);
}


template<typename _T>
inline
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         std::vector<_T>* receive, MPI_Comm const comm)
{
  // Check that the number of arrays matches the number of processes.
  std::size_t const numProcesses = send.numArrays();
  assert(numProcesses == std::size_t(mpi::commSize(comm)));

  // Record the number of objects that we will send to each process.
  std::vector<int> sendCounts(numProcesses);
  for (std::size_t i = 0; i != sendCounts.size(); ++i) {
    sendCounts[i] = send.size(i);
  }
  // Record the send displacements.
  std::vector<int> sendDisplacements(numProcesses);
  for (std::size_t i = 0; i != sendDisplacements.size(); ++i) {
    sendDisplacements[i] = send.begin(i) - send.begin();
  }

  // Determine the number of objects that we will receive from each process.
  std::vector<int> receiveCounts(numProcesses);
  allToAll(sendCounts, &receiveCounts, comm);
  // Calculate the receive displacements.
  std::vector<int> receiveDisplacements(numProcesses);
  receiveDisplacements[0] = 0;
  for (std::size_t i = 1; i != receiveDisplacements.size(); ++i) {
    receiveDisplacements[i] = receiveDisplacements[i - 1] +
      receiveCounts[i - 1];
  }
  // Allocate memory for the objects that we will receive.
  receive->resize(ext::sum(receiveCounts));

  // Perform the all-to-all communication.
  Data<_T> const data;
  allToAll(send.data(), &sendCounts[0], &sendDisplacements[0], data.type(),
           &receive->front(), &receiveCounts[0], &receiveDisplacements[0],
           data.type(), comm);
}


template<typename _T>
inline
void
allToAll(std::vector<_T> const& sendBuf,
         std::vector<std::size_t> const& sendCounts,
         std::vector<_T>* recvBuf, MPI_Comm const comm)
{
  assert(sendCounts.size() == std::size_t(mpi::commSize(comm)));

  // Convert the send counts to int.
  std::vector<int> sendCountsInt(sendCounts.size());
  for (std::size_t i = 0; i != sendCountsInt.size(); ++i) {
    sendCountsInt[i] = sendCounts[i];
  }
  // Note that we are also verifying that int is sufficient for counting
  // the size.
  assert(std::size_t(ext::sum(sendCountsInt)) == sendBuf.size());

  // Perform an all-to-all communication to determine the number of objects
  // that we will receive.
  std::vector<int> recvCounts(sendCountsInt.size());
  allToAll(sendCountsInt, &recvCounts, comm);

  // Allocate memory for the receive buffer.
  recvBuf->resize(ext::sum(recvCounts));

  // Exchange the data.
  Data<_T> const data;
  allToAll(&sendBuf[0], &sendCountsInt[0], data.type(),
           &(*recvBuf)[0], &recvCounts[0], data.type(), comm);
}


template<typename _T>
inline
void
allToAll(std::vector<_T> const& send, std::vector<_T>* receive, MPI_Comm comm)
{
#ifdef STLIB_DEBUG
  assert(send.size() == std::size_t(commSize(comm)));
#endif
  assert(send.size() == receive->size());
  // Perform the communication.
  Data<_T> const data;
  allToAll(&send.front(), 1, data.type(), &receive->front(), 1, data.type(),
           comm);
}


inline
void
allToAll(void const* const sendbuf, int const sendcount,
         MPI_Datatype const sendtype, void* const recvbuf, int const recvcount,
         MPI_Datatype const recvtype, MPI_Comm const comm)
{
#if MPI_VERSION >= 3
  void const* const sendbuf_ = sendbuf;
#else
  void* const sendbuf_ = const_cast<void*>(sendbuf);
#endif

  if (MPI_Alltoall(sendbuf_, sendcount, sendtype, recvbuf, recvcount, recvtype,
                   comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Alltoall is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
void
allToAll(_T const* send, int const* sendCounts, _T* recv, int const* recvCounts,
         MPI_Comm comm)
{
  Data<_T> const data;
  allToAll(send, sendCounts, data.type(), recv, recvCounts, data.type(), comm);
}


inline
void
allToAll(void const* const sendbuf, int const* const sendcounts,
         MPI_Datatype const sendtype,
         void* const recvbuf, int const* const recvcounts,
         MPI_Datatype const recvtype,
         MPI_Comm const comm)
{
  std::size_t const commSize = mpi::commSize(comm);
  // Calculate displacements from the send/receive counts.
  std::vector<int> sendDisplacements(commSize);
  sendDisplacements[0] = 0;
  for (std::size_t i = 1; i != sendDisplacements.size(); ++i) {
    sendDisplacements[i] = sendDisplacements[i - 1] +
      sendcounts[i - 1];
  }
  std::vector<int> recvDisplacements(commSize);
  recvDisplacements[0] = 0;
  for (std::size_t i = 1; i != recvDisplacements.size(); ++i) {
    recvDisplacements[i] = recvDisplacements[i - 1] +
      recvcounts[i - 1];
  }
  // Perform the all-to-all communication.
  allToAll(sendbuf, sendcounts, &sendDisplacements[0], sendtype,
           recvbuf, recvcounts, &recvDisplacements[0], recvtype, comm);
}


inline
void
allToAll(void const* const sendbuf, int const* const sendcounts,
         int const* const sdispls, MPI_Datatype const sendtype,
         void* const recvbuf, int const* const recvcounts,
         int const* const rdispls, MPI_Datatype const recvtype,
         MPI_Comm const comm)
{
#if MPI_VERSION >= 3
  void const* const sendbuf_ = sendbuf;
  int const* const sendcounts_ = sendcounts;
  int const* const sdispls_ = sdispls;
  int const* const recvcounts_ = recvcounts;
  int const* const rdispls_ = rdispls;
#else
  void* const sendbuf_ = const_cast<void*>(sendbuf);
  int* const sendcounts_ = const_cast<int*>(sendcounts);
  int* const sdispls_ = const_cast<int*>(sdispls);
  int* const recvcounts_ = const_cast<int*>(recvcounts);
  int* const rdispls_ = const_cast<int*>(rdispls);
#endif
  
  if (MPI_Alltoallv(sendbuf_, sendcounts_, sdispls_, sendtype, recvbuf,
                    recvcounts_, rdispls_, recvtype, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Alltoallv is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         std::vector<_T>* receive,
         int const tag,
         std::vector<MPI_Request>* sendRequests,
         std::vector<MPI_Request>* receiveRequests,
         MPI_Comm comm)
{
  // Check that the number of arrays matches the number of processes.
  std::size_t const numProcesses = send.numArrays();
  assert(numProcesses == std::size_t(mpi::commSize(comm)));

  // Record the number of objects that we will send to each process.
  std::vector<std::size_t> sendCounts(numProcesses);
  for (std::size_t i = 0; i != sendCounts.size(); ++i) {
    sendCounts[i] = send.size(i);
  }
  // Record the send displacements.
  std::vector<std::size_t> sendDisplacements(numProcesses);
  for (std::size_t i = 0; i != sendDisplacements.size(); ++i) {
    sendDisplacements[i] = send.begin(i) - send.begin();
  }

  // Determine the number of objects that we will receive from each process.
  std::vector<std::size_t> receiveCounts(numProcesses);
  allToAll(sendCounts, &receiveCounts, comm);
  // Calculate the receive displacements.
  std::vector<std::size_t> receiveDisplacements(numProcesses);
  receiveDisplacements[0] = 0;
  for (std::size_t i = 1; i != receiveDisplacements.size(); ++i) {
    receiveDisplacements[i] = receiveDisplacements[i - 1] +
      receiveCounts[i - 1];
  }
  // Allocate memory for the objects that we will receive.
  receive->resize(ext::sum(receiveCounts));

  // Allocate memory for the receive requests.
  {
    std::size_t numReceives = 0;
    for (std::size_t i = 0; i != receiveCounts.size(); ++i) {
      if (receiveCounts[i]) {
        ++numReceives;
      }
    }
    receiveRequests->resize(numReceives);
  }
  // Initiate receives.
  {
    std::size_t n = 0;
    for (std::size_t i = 0; i != receiveCounts.size(); ++i) {
      // If we are receiving objects from the i_th process.
      if (receiveCounts[i]) {
        (*receiveRequests)[n++] =
          iRecv(&(*receive)[receiveDisplacements[i]], receiveCounts[i], i, tag,
                comm);
      }
    }
  }

  // Allocate memory for the send requests.
  {
    std::size_t numSends = 0;
    for (std::size_t i = 0; i != sendCounts.size(); ++i) {
      if (sendCounts[i]) {
        ++numSends;
      }
    }
    sendRequests->resize(numSends);
  }
  // Initiate receives.
  {
    std::size_t n = 0;
    for (std::size_t i = 0; i != sendCounts.size(); ++i) {
      // If we are sending objects to the i_th process.
      if (sendCounts[i]) {
        (*sendRequests)[n++] = iSend(&send(i, 0), sendCounts[i], i, tag, comm);
      }
    }
  }
}


} // namespace mpi
} // namespace stlib

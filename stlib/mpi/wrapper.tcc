// -*- C++ -*-

#if !defined(__mpi_wrapper_tcc__)
#error This file is an implementation detail of wrapper.
#endif

#if 0
// This error checking macro is used only in this file.
#define _MPI_CHECK_ERROR(EX) int __errorCode = (EX); \
  (void)((EX) == MPI_SUCCESS || \
         (_checkError(#EX, __errorCode, __FILE__, __LINE__), 0))
#endif

namespace stlib
{
namespace mpi
{


#if 0
inline
void
_checkError(char const* call, int const code, char const* file, int const line)
{
#if 0
  // Using ostringstream causes a segfault. I don't know why.
  std::ostringstream stream;
  stream << "The call, \"" << call
         << "\" on line " << line
         << " of " << file
         << " returned " << code
         << " instead of MPI_SUCCESS.";
  throw std::runtime_error(stream.str());
#else
  std::cerr << "The call, \"" << call
            << "\" on line " << line
            << " of " << file
            << " returned " << code
            << " instead of MPI_SUCCESS.";
  throw std::runtime_error(call);
#endif
}
#endif


template<typename _Integer>
inline
std::vector<_Integer>
calculateDisplacements(std::vector<_Integer> const& counts)
{
  // The group size may not be zero.
  assert(! counts.empty());
  std::vector<_Integer> displacements(counts.size());
  displacements[0] = 0;
  for (std::size_t i = 1; i != displacements.size(); ++i) {
    displacements[i] = displacements[i - 1] + counts[i - 1];
  }
  return displacements;
}


inline
int
commSize(MPI_Comm comm)
{
  int size = 0;
  if (MPI_Comm_size(comm, &size) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Comm_size is not MPI_SUCCESS.");
  }
  return size;
}


inline
int
commRank(MPI_Comm comm)
{
  int rank = 0;
  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Comm_rank is not MPI_SUCCESS.");
  }
  return rank;
}


inline
void
barrier(MPI_Comm comm)
{
  if (MPI_Barrier(comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Barrier is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
void
send(std::vector<_T> const& input, int const dest, int const tag,
     MPI_Comm const comm)
{
  Data<_T> const data;
  send(&input[0], input.size(), data.type(), dest, tag, comm);
}


template<typename _T>
inline
std::vector<_T>
recv(int const source, int const tag, MPI_Comm const comm)
{
  std::vector<_T> output;
  recv(&output, source, tag, comm);
  return output;
}


template<typename _T>
inline
void
recv(std::vector<_T>* output, int const source, int const tag,
     MPI_Comm const comm)
{
  // First probe to get the size of the buffer.
  MPI_Status status = probe(source, tag, comm);
  Data<_T> const data;
  int const count = getCount(status, data.type());
  // Allocate memory.
  output->resize(count);
  // Receive the buffer. Note that we must use status.MPI_SOURCE because 
  // the source parameter may be MPI_ANY_SOURCE.
  recv(&(*output)[0], output->size(), data.type(), status.MPI_SOURCE, tag,
       comm);
}


inline
void
send(void const* const buf, std::size_t const count,
     MPI_Datatype const datatype, int const dest, int const tag,
     MPI_Comm const comm)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Send exceeds the "
                              "capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = buf;
#else
  void* const buffer = const_cast<void*>(buf);
#endif
  if (MPI_Send(buffer, count, datatype, dest, tag, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Send is not MPI_SUCCESS.");
  }
}


inline
void
recv(void* const buf, std::size_t const count, MPI_Datatype const datatype,
     int const source, int const tag, MPI_Comm const comm)
{
  MPI_Status status;
  recv(buf, count, datatype, source, tag, &status, comm);
  int const c = getCount(status, datatype);
  if (std::size_t(c) != count) {
    std::ostringstream stream;
    stream << "Expected to receive " << count << " objects in MPI_Recv, but "
           << "received " << c << " instead.";
    throw std::runtime_error(stream.str());
  }
}


inline
void
recv(void* const buf, std::size_t const count, MPI_Datatype const datatype,
     int const source, int const tag, MPI_Status* const status,
     MPI_Comm const comm)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Recv exceeds the "
                              "capacity of the int type.");
  }
  if (MPI_Recv(buf, count, datatype, source, tag, comm, status) !=
      MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Recv is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
void
sendRecv(_T const& send, _T *recv, int const rank, int const tag,
         MPI_Comm const comm)
{
  sendRecv(send, rank, tag, recv, rank, tag, comm);
}


template<typename _T>
inline
void
sendRecv(_T const& send, int const dest, int const sendTag,
         _T *recv, int const source, int const recvTag, MPI_Comm const comm)
{
  Data<_T> const data;
  sendRecv(send, data.type(), dest, sendTag,
           recv, data.type(), source, recvTag, comm);
}


template<typename _T>
inline
void
sendRecv(_T const& send, MPI_Datatype const sendType, int const dest,
         int const sendTag, _T *recv, MPI_Datatype const recvType,
         int const source, int const recvTag, MPI_Comm const comm)
{
  sendRecv(&send, 1, sendType, dest, sendTag,
           recv, 1, recvType, source, recvTag, comm);
}


template<typename _T>
inline
void
sendRecv(std::vector<_T> const& send, int const dest,
         std::vector<_T>* const recv, int const source, int const tag,
         MPI_Comm const comm)
{
  sendRecv(&send[0], send.size(), dest, &(*recv)[0], recv->size(), source, tag,
           comm);
}


template<typename _T>
inline
void
sendRecv(_T const* sendBuf, std::size_t const sendCount,
         int const dest,
         _T* recvBuf, std::size_t const recvCount,
         int const source, int const tag,
         MPI_Comm const comm)
{
  sendRecv(sendBuf, sendCount, dest, tag,
           recvBuf, recvCount, source, tag, comm);
}


template<typename _T>
inline
void
sendRecv(_T const* sendBuf, std::size_t const sendCount,
         int const dest, int const sendTag,
         _T* recvBuf, std::size_t const recvCount,
         int const source, int const recvTag,
         MPI_Comm const comm)
{
  Data<_T> const data;
  sendRecv(sendBuf, sendCount, data.type(), dest, sendTag,
           recvBuf, recvCount, data.type(), source, recvTag, comm);
}


inline
void
sendRecv(void const* sendBuf, std::size_t const sendCount,
         MPI_Datatype const sendType, int const dest, int const sendTag,
         void *recvBuf, std::size_t const recvCount,
         MPI_Datatype const recvType, int const source, int const recvTag,
         MPI_Comm const comm)
{
  if (sendCount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error
      ("The send count argument to MPI_Sendrecv exceeds the capacity of the "
       "int type.");
  }
  if (recvCount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error
      ("The receive count argument to MPI_Sendrecv exceeds the capacity of the "
       "int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = sendBuf;
#else
  void* const buffer = const_cast<void*>(sendBuf);
#endif
  MPI_Status status;
#if 0
  std::cerr << "sendCount = " << sendCount << '\n'
            << "dest = " << dest << '\n'
            << "sendTag = " << sendTag << '\n'
            << "recvCount = " << recvCount << '\n'
            << "source = " << source << '\n'
            << "recvTag = " << recvTag << '\n' << '\n';
#endif
  if (MPI_Sendrecv(buffer, sendCount, sendType, dest, sendTag, recvBuf,
                   recvCount, recvType, source, recvTag, comm, &status) !=
      MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Sendrecv is not MPI_SUCCESS.");
  }
  int const c = getCount(status, recvType);
  if (std::size_t(c) != recvCount) {
    std::ostringstream stream;
    stream << "Expected to receive " << recvCount
           << " objects in MPI_Sendrecv, but received " << c << " instead.";
    throw std::runtime_error(stream.str());
  }
}




template<typename _T>
inline
MPI_Request
iSend(std::vector<_T> const& buf, int const dest, int const tag,
      MPI_Comm const comm)
{
  return iSend(&buf[0], buf.size(), dest, tag, comm);
}


template<typename _T>
inline
MPI_Request
iRecv(std::vector<_T>* buf, int const source, int const tag,
      MPI_Comm const comm)
{
  return iRecv(&(*buf)[0], buf->size(), source, tag, comm);
}


template<typename _T>
inline
MPI_Request
iSend(_T const* const buf, std::size_t const count, int const dest,
      int const tag, MPI_Comm const comm)
{
  Data<_T> const data;
  return iSend(buf, count, data.type(), dest, tag, comm);
}


template<typename _T>
inline
MPI_Request
iRecv(_T* const buf, std::size_t const count, int const source, int const tag,
      MPI_Comm const comm)
{
  Data<_T> const data;
  return iRecv(buf, count, data.type(), source, tag, comm);
}


inline
MPI_Request
iSend(void const* const buf, std::size_t const count,
      MPI_Datatype const datatype, int const dest,
      int const tag, MPI_Comm const comm)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Isend exceeds the "
                              "capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const _buf = buf;
#else
  void* const _buf = const_cast<void*>(buf);
#endif
  MPI_Request request;
  if (MPI_Isend(_buf, count, datatype, dest, tag, comm, &request) !=
      MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Isend is not MPI_SUCCESS.");
  }
  return request;
}


inline
MPI_Request
iRecv(void* const buf, std::size_t const count, MPI_Datatype const datatype,
      int const source, int const tag, MPI_Comm const comm)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_IRecv exceeds the "
                              "capacity of the int type.");
  }
  MPI_Request request;
  if (MPI_Irecv(buf, count, datatype, source, tag, comm, &request) !=
      MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Irecv is not MPI_SUCCESS.");
  }
  return request;
}


inline
MPI_Status
probe(int const source, int const tag, MPI_Comm const comm)
{
  MPI_Status status;
  probe(source, tag, comm, &status);
  return status;
}


inline
void
probe(int const source, int const tag, MPI_Comm const comm,
      MPI_Status* const status)
{
  if (MPI_Probe(source, tag, comm, status) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Probe() is not MPI_SUCCESS.");
  }
}


inline
void
wait(MPI_Request *request)
{
  if (MPI_Wait(request, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Wait() is not MPI_SUCCESS.");
  }
}


inline
void
wait(MPI_Request* request, MPI_Status* status)
{
  if (MPI_Wait(request, status) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Wait() is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
void
wait(MPI_Request* request, std::size_t const count)
{
  MPI_Status status;
  if (MPI_Wait(request, &status) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Wait() is not MPI_SUCCESS.");
  }
  int const c = getCount<_T>(status);
  if (std::size_t(c) != count) {
    std::ostringstream stream;
    stream << "Expected to receive " << count << " objects, but received "
           << c << " instead.";
    throw std::runtime_error(stream.str());
  }
}


inline
void
waitAll(std::vector<MPI_Request>* requests)
{
  waitAll(requests->size(), &requests->front(), MPI_STATUSES_IGNORE);
}


inline
void
waitAll(std::vector<MPI_Request>* requests, std::vector<MPI_Status>* statuses)
{
  assert(requests->size() == statuses->size());
  waitAll(requests->size(), &requests->front(), &statuses->front());
}


inline
void
waitAll(int const count, MPI_Request* requests, MPI_Status* statuses)
{
  if (MPI_Waitall(count, requests, statuses) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Waitall is not MPI_SUCCESS.");
  }
}


inline
int
getCount(MPI_Status const& status, MPI_Datatype const datatype)
{
  int count = 0;
#if MPI_VERSION >= 3
  MPI_Status const* const _status = &status;
#else
  MPI_Status* const _status = const_cast<MPI_Status*>(&status);
#endif
  if (MPI_Get_count(_status, datatype, &count) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Get_count is not MPI_SUCCESS.");
  }
  return count;
}


template<typename _T>
inline
int
getCount(MPI_Status const& status)
{
  Data<_T> const data;
  return getCount(status, data.type());
}




template<typename _T>
inline
std::vector<_T>
gather(_T const& input, MPI_Comm const comm, int const root)
{
  std::vector<_T> output;
  if (commRank(comm) == root) {
    output.resize(commSize(comm));
  }
  Data<_T> const data;
  gather(&input, 1, data.type(), &output.front(), 1, data.type(), comm, root);
  return output;
}


inline
void
gather(void const* const sendbuf, std::size_t const sendcount,
       MPI_Datatype const sendtype, void* const recvbuf,
       std::size_t const recvcount, MPI_Datatype const recvtype,
       MPI_Comm const comm, int const root)
{
  if (sendcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sendcount argument to MPI_Gather exceeds "
                              "the capacity of the int type.");
  }
  if (recvcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The recvcount argument to MPI_Gather exceeds "
                              "the capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = sendbuf;
#else
  void* const buffer = const_cast<void*>(sendbuf);
#endif
  if (MPI_Gather(buffer, sendcount, sendtype, recvbuf, recvcount, recvtype,
                 root, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Gather is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
std::vector<_T>
gather(std::vector<_T> const& send, MPI_Comm const comm, int const root)
{
  if (send.size() > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sent vector size in mpi::gather() exceeds "
                              "the capacity of the int type.");
  }
  // Gather the send counts to the root.
  std::vector<int> const recvCounts = gather(int(send.size()), comm, root);

  std::vector<_T> recv;
  std::vector<int> displacements;
  if (commRank(comm) == root) {
    // Check for overflow.
    std::size_t total = 0;
    for (std::size_t i = 0; i != recvCounts.size(); ++i) {
      total += recvCounts[i];
    }
    if (total > std::size_t(std::numeric_limits<int>::max())) {
      throw std::overflow_error("The received vector size in mpi::gather() "
                                "exceeds the capacity of the int type.");
    }
    // Allocate memory.
    recv.resize(total);
    // Calculate the displacements.
    displacements = calculateDisplacements(recvCounts);
  }
  
  // Perform the gather.
  gather(&send[0], send.size(), &recv[0], &recvCounts[0], &displacements[0],
         comm, root);
  return recv;
}


template<typename _T>
inline
void
gather(_T const* const sendbuf, int const sendcount, _T* const recvbuf,
       int const* const recvcounts, int const* const displs,
       MPI_Comm const comm, int const root)
{
  Data<_T> const data;
  gather(sendbuf, sendcount, data.type(), recvbuf, recvcounts, displs,
         data.type(), comm, root);
}


inline
void
gather(void const* const sendbuf, int const sendcount,
       MPI_Datatype const sendtype, void* const recvbuf,
       int const* const recvcounts, int const* const displs,
       MPI_Datatype const recvtype, MPI_Comm const comm, int const root)
{
#if MPI_VERSION >= 3
  void const* const _sendbuf = sendbuf;
  int const* const _recvcounts = recvcounts;
  int const* const _displs = displs;
#else
  void* const _sendbuf = const_cast<void*>(sendbuf);
  int* const _recvcounts = const_cast<int*>(recvcounts);
  int* const _displs = const_cast<int*>(displs);
#endif
  if (MPI_Gatherv(_sendbuf, sendcount, sendtype, recvbuf, _recvcounts, _displs,
                  recvtype, root, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Gatherv is not MPI_SUCCESS.");
  }
  
}


template<typename _T>
inline
std::vector<_T>
allGather(_T const& input, MPI_Comm const comm)
{
  std::vector<_T> output(commSize(comm));
  Data<_T> const data;
  allGather(&input, 1, data.type(), &output.front(), 1, data.type(), comm);
  return output;
}


inline
void
allGather(void const* const sendbuf, std::size_t const sendcount,
          MPI_Datatype const sendtype, void* const recvbuf,
          std::size_t const recvcount, MPI_Datatype const recvtype,
          MPI_Comm const comm)
{
  if (sendcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sendcount argument to MPI_Allgather exceeds "
                              "the capacity of the int type.");
  }
  if (recvcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The recvcount argument to MPI_Allgather exceeds "
                              "the capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = sendbuf;
#else
  void* const buffer = const_cast<void*>(sendbuf);
#endif
  if (MPI_Allgather(buffer, sendcount, sendtype, recvbuf, recvcount, recvtype,
                    comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Allgather is not MPI_SUCCESS.");
  }
}


template<typename _T>
inline
std::vector<_T>
allGather(std::vector<_T> const& sendBuf, MPI_Comm const comm)
{
  std::size_t const commSize = mpi::commSize(comm);
  if (sendBuf.size() > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sendcount argument to MPI_Allgatherv "
                              "exceeds the capacity of the int type.");
  }
  // Gather the receive counts.
  std::vector<int> const recvcounts = allGather(int(sendBuf.size()), comm);
  // Calculate the displacements.
  std::vector<int> displs(commSize);
  displs.front() = 0;
  for (std::size_t i = 1; i != displs.size(); ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }
  // Allocate memory for the receive buffer.
  std::vector<_T> recvBuf(displs.back() + recvcounts.back());
  // Perform the allGather.
  Data<_T> const data;
  mpi::allGather(&sendBuf.front(), sendBuf.size(), data.type(),
                 &recvBuf.front(), &recvcounts.front(), &displs.front(),
                 data.type(), comm);
  return recvBuf;
}


template<typename _T>
inline
container::PackedArrayOfArrays<_T>
allGatherPacked(std::vector<_T> const& send, MPI_Comm const comm)
{
  std::size_t const commSize = mpi::commSize(comm);
  if (send.size() > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sendcount argument to MPI_Allgatherv "
                              "exceeds the capacity of the int type.");
  }
  // Gather the receive counts.
  std::vector<int> const recvcounts = allGather(int(send.size()), comm);
  // Calculate the displacements.
  std::vector<int> displs(commSize);
  displs.front() = 0;
  for (std::size_t i = 1; i != displs.size(); ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }
  // Allocate memory for the objects we will receive.
  container::PackedArrayOfArrays<_T> receive;
  receive.rebuild(recvcounts.begin(), recvcounts.end());
  // Perform the allGather.
  Data<_T> const data;
  mpi::allGather(&send.front(), send.size(), data.type(),
                 receive.data(), &recvcounts.front(), &displs.front(),
                 data.type(), comm);
  return receive;
}


inline
void
allGather(void const* const sendbuf, int const sendcount,
          MPI_Datatype const sendtype, void* const recvbuf,
          int const* const recvcounts, int const* const displs,
          MPI_Datatype const recvtype, MPI_Comm const comm)
{
#if MPI_VERSION >= 3
  void const* const _sendbuf = sendbuf;
  int const* const _recvcounts = recvcounts;
  int const* const _displs = displs;
#else
  void* const _sendbuf = const_cast<void*>(sendbuf);
  int* const _recvcounts = const_cast<int*>(recvcounts);
  int* const _displs = const_cast<int*>(displs);
#endif
  if (MPI_Allgatherv(_sendbuf, sendcount, sendtype, recvbuf, _recvcounts,
                     _displs, recvtype, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Allgatherv is not "
                             "MPI_SUCCESS.");
  }
}




template<typename _T>
inline
_T
reduce(_T const object, MPI_Op const op, MPI_Comm const comm, int const root)
{
  _T reduced;
  Data<_T> const data;
  reduce(&object, &reduced, 1, data.type(), op, comm, root);
  return reduced;
}


template<typename _T>
inline
void
reduce(std::vector<_T> const& sendObjects, std::vector<_T>* const recvObjects,
       MPI_Op const op, MPI_Comm const comm, int const root)
{
  // The send vector may not be the same as the receive vector because the
  // latter will be resized before the communication.
  assert(&sendObjects != recvObjects);

  if (commRank(comm) == root) {
    recvObjects->resize(sendObjects.size());
  }
  Data<_T> const data;
  reduce(&sendObjects.front(), &recvObjects->front(), sendObjects.size(),
         data.type(), op, comm, root);
}


inline
void
reduce(void const* const sendbuf, void* const recvbuf,
       std::size_t const count, MPI_Datatype const datatype,
       MPI_Op const op, MPI_Comm const comm, int const root)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Reduce exceeds the "
                              "capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = sendbuf;
#else
  void* const buffer = const_cast<void*>(sendbuf);
#endif
  MPI_Reduce(buffer, recvbuf, count, datatype, op, root, comm);
}




inline
void
allReduce(void const* const sendbuf, void* const recvbuf,
          std::size_t const count, MPI_Datatype const datatype,
          MPI_Op const op, MPI_Comm const comm)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Allreduce exceeds the "
                              "capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const buffer = sendbuf;
#else
  void* const buffer = const_cast<void*>(sendbuf);
#endif
  MPI_Allreduce(buffer, recvbuf, count, datatype, op, comm);
}


template<typename _T>
inline
_T
allReduceSum(_T const object, MPI_Comm const comm)
{
  return allReduce(object, MPI_SUM, comm);
}


template<>
inline
std::size_t
allReduceSum(std::size_t const object, MPI_Comm const comm)
{
  // Note that we don't support sizes that require extended
  // precision on 32-bit systems, but we do check for problems.
  unsigned long long const total =
    mpi::allReduce((unsigned long long)(object), MPI_SUM, comm);
  if (total > std::size_t(std::numeric_limits<std::size_t>::max())) {
    std::ostringstream stream;
    stream << "The total number of distributed objects, " << total
           << " exceeds the capacity of std::size_t. Run this calculation on "
           << "a 64-bit system instead.";
    throw std::overflow_error(stream.str());
  }
  return total;
}


template<typename _T>
inline
void
allReduce(std::vector<_T> const& sendObjects,
          std::vector<_T>* const recvObjects, MPI_Op const op,
          MPI_Comm const comm)
{
  if (sendObjects.size() != recvObjects->size()) {
    throw std::runtime_error("In mpi::allReduce(), the sizes of the send and "
                             "receive vectors do not match.");
  }
  Data<_T> const data;
  allReduce(&sendObjects.front(), &recvObjects->front(), sendObjects.size(),
            data.type(), op, comm);
}


template<typename _T>
inline
_T
allReduce(_T const object, MPI_Op const op, MPI_Comm const comm)
{
  _T reduced;
  Data<_T> const data;
  allReduce(&object, &reduced, 1, data.type(), op, comm);
  return reduced;
}




template<typename _T>
inline
void
bcast(_T* const object, MPI_Comm const comm, int const root)
{
  bcast(object, 1, comm, root);
}


template<typename _T>
void
bcast(std::vector<_T>* objects, MPI_Comm const comm, int const root)
{
  if (objects->size() > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The std::vector is too large to be broadcast "
                              "with MPI_Bcast().");
  }
  // First broadcast the count.
  int const rank = commRank(comm);
  std::size_t count = 0;
  if (rank == root) {
    count = objects->size();
  }
  bcast(&count, comm);
  if (rank != root) {
    objects->resize(count);
  }
  // Then broadcast the objects.
  bcast(&objects->front(), count, comm, root);
}


template<typename _T>
void
bcastNoResize(std::vector<_T>* objects, MPI_Comm const comm, int const root)
{
  if (objects->size() > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The std::vector is too large to be broadcast "
                              "with MPI_Bcast().");
  }
  bcast(&objects->front(), objects->size(), comm, root);
}


template<typename _T>
inline
void
bcast(_T* const buffer, std::size_t const count, MPI_Comm const comm,
      int const root)
{
  Data<_T> const data;
  bcast(buffer, count, data.type(), comm, root);
}


inline
void
bcast(void* const buffer, std::size_t const count, MPI_Datatype const datatype,
      MPI_Comm const comm, int const root)
{
  if (count > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The count argument to MPI_Bcast exceeds the "
                              "capacity of the int type.");
  }
  MPI_Bcast(buffer, count, datatype, root, comm);
}


template<typename _T>
inline
_T
scatter(std::vector<_T> const& send, MPI_Comm const comm, int const root)
{
  if (commRank(comm) == root) {
    assert(std::size_t(commSize(comm)) == send.size());
  }
  _T result;
  scatter(&send[0], 1, &result, comm, root);
  return result;
}


template<typename _T>
inline
void
scatter(_T const* sendbuf, std::size_t const count, _T* recvbuf,
        MPI_Comm const comm, int const root)
{
  Data<_T> const data;
  scatter(sendbuf, count, data.type(), recvbuf, count, data.type(), comm, root);
}


inline
void
scatter(void const* const sendbuf, std::size_t const sendcount,
        MPI_Datatype const sendtype, void* const recvbuf,
        std::size_t const recvcount, MPI_Datatype const recvtype,
        MPI_Comm const comm, int const root)
{
  if (sendcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The sendcount argument to MPI_Scatter exceeds "
                              "the capacity of the int type.");
  }
  if (recvcount > std::size_t(std::numeric_limits<int>::max())) {
    throw std::overflow_error("The recvcount argument to MPI_Scatter exceeds "
                              "the capacity of the int type.");
  }
#if MPI_VERSION >= 3
  void const* const sendbuf_ = sendbuf;
#else
  void* const sendbuf_ = const_cast<void*>(sendbuf);
#endif
  if (MPI_Scatter(sendbuf_, sendcount, sendtype, recvbuf, recvcount, recvtype,
                  root, comm) != MPI_SUCCESS) {
    throw std::runtime_error("The result of MPI_Scatter is not MPI_SUCCESS.");
  }
}


} // namespace mpi
} // namespace stlib

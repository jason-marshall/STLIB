// -*- C++ -*-

/*!
  \file mpi/wrapper.h
  \brief Wrappers for MPI functions.

  Some functions are convenience wrappers. Others check for overflow or
  correct defects in various MPI implementations. Note that for the functions
  that take the root parameter, this parameter is last. (This is different
  than the parameter order for MPI functions.) This is done to simplify
  the interface; root has a default value of 0.
*/

#if !defined(__mpi_wrapper_h__)
#define __mpi_wrapper_h__

#include "stlib/mpi/Data.h"

#include "stlib/container/PackedArrayOfArrays.h"

#include <limits>
#include <sstream>
#include <vector>


namespace stlib
{
namespace mpi
{


/** \defgroup mpiWrapper Wrappers for MPI functions
    Functions that provide a more convenient interface and check errors.
    @{
*/


/** Calculate displacements from the counts.
This assumes that objects being sent or received are contiguous and not 
overlapping, which is usually the case.
*/
template<typename _Integer>
void
calculateDisplacements(std::vector<_Integer> const& counts,
                       std::vector<_Integer>* displacements);


/** \defgroup mpiWrapperSizeRank MPI_Commsize() and MPI_Commrank()
    These functions return the size and rank instead of relying on passing 
    a pointer. This help avoid problems with unused variables. Below is an 
    example usage.

    \code
    int const commSize = mpi::commSize(comm);
    \endcode

    @{
*/

/// Return the size of the group associated with a communicator.
int
commSize(MPI_Comm comm = MPI_COMM_WORLD);

/// Return the rank of the calling process in the communicator.
int
commRank(MPI_Comm comm = MPI_COMM_WORLD);

/// Blocks until all processes in the communicator have reached this routine.
void
barrier(MPI_Comm comm = MPI_COMM_WORLD);

/** @} */ // End of mpiWrapperSizeRank group.



/** \defgroup mpiWrapperSendRecv MPI_Send(), MPI_Recv(), and MPI_Sendrecv()
    The simplest interface for sending and receiving uses \c std::vector. 
    If the MPI data type can be deduced using the Data class, then follow
    the example below. For sending, the count is deduced from the %container
    size. For receiving, the vector will be automatically resized to the
    length of the message.
    \code
    std::vector<float> sendData(size);
    ...
    mpi::send(&sendData, dest, tag, comm);
    \endcode
    \code
    std::vector<float> recvData;
    mpi::recv(&recvData, source, tag, comm);
    \endcode
    For the version that returns the vector of objects, you must specify
    the object type.
    \code
    std::vector<float> recvData = mpi::recv<float>(source, tag, comm);
    \endcode
    
    If you data is not in \c std::vector's, then the wrapper only checks for
    count overflow and MPI errors.
    \code
    Foo sendBuffer[count];
    ...
    mpi::send(sendBuffer, count, fooDataType, dest, tag, comm);
    \endcode
    \code
    Foo recvBuffer[count];
    mpi::recv(recvBuffer, count, fooDataType, source, tag, comm);
    \endcode

    There are wrappers for \c MPI_Sendrecv() that check for errors and 
    provide a simpler interface. In the simplest case, you can send and receive
    an object to and from the same rank.
    \code
    Foo send;
    ...
    Foo recv;
    mpi::sendRecv(send, &recv, rank, tag, comm);
    \endcode
    If the destination and source ranks are the same, you specify them.
    \code
    mpi::sendRecv(send, dest, sendTag, &recv, source, recvTag, comm);
    \endcode
    If the MPI data type cannot be deduced, you will need to specify it.
    \code
    mpi::sendRecv(send, sendType, dest, sendTag, &recv, recvType, source, recvTag, comm);
    \endcode
    Finally, there is a wrapper that just checks for errors.
    @{
*/

/// Wrapper for sending a \c std::vector with MPI_Send(). Deduce the data type.
/**
  \param input The vector to send.
  \param dest Rank of destination (integer)
  \param tag Message tag (integer)
  \param comm Communicator (handle)
*/
template<typename _T>
void
send(std::vector<_T> const& input, int dest, int tag,
     MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for receiving a \c std::vector with MPI_Recv(). Deduce the data type.
/**
  \param source Rank of source.
  \param tag Message tag.
  \param comm Communicator.
  \return The vector of received objects.
*/
template<typename _T>
std::vector<_T>
recv(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for receiving a \c std::vector with MPI_Recv(). Deduce the data type.
/**
  \param output The vector to receive. The vector is resized after 
  probing the message and before receiving it.
  \param source Rank of source.
  \param tag Message tag.
  \param comm Communicator.
*/
template<typename _T>
void
recv(std::vector<_T>* output, int source, int tag,
     MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Send() that checks for overflow in the \c count argument.
/**
  \param buf Initial address of send buffer (choice)
  \param count Number of elements in send buffer. Note that this is specified
  using \c std::size_t. The conversion to \c int will be checked.
  \param datatype %Data type of each send buffer element (handle)
  \param dest Rank of destination (integer)
  \param tag Message tag (integer)
  \param comm Communicator (handle)
*/
void
send(void const* buf, std::size_t count, MPI_Datatype datatype, int dest,
     int tag, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Recv() that checks for overflow in the \c count argument.
/**
  \param buf Initial address of receive buffer (choice)
  \param count Exact number of elements in receive buffer. Note that this is
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param datatype %Data type of each receive buffer element (handle)
  \param source Rank of source (integer)
  \param tag Message tag (integer)
  \param comm Communicator (handle)
*/
void
recv(void *buf, std::size_t count, MPI_Datatype datatype, int source, int tag,
     MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Recv() that checks for overflow in the \c count argument.
/**
  \param buf Initial address of receive buffer.
  \param count Maximum number of elements in receive buffer. Note that this is
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param datatype %Data type of each receive buffer element.
  \param source Rank of source.
  \param tag Message tag.
  \param status The status struct.
  \param comm Communicator.

  Check the return code.
*/
void
recv(void *buf, std::size_t count, MPI_Datatype datatype, int source, int tag,
     MPI_Status* status, MPI_Comm comm = MPI_COMM_WORLD);

/// Send and receive a single object from the same rank. Deduce the data type.
/**
  \param send The object to send.
  \param recv The object to receive.
  \param rank Rank of source and destination.
  \param tag Message tag.
  \param comm Communicator.
*/
template<typename _T>
void
sendRecv(_T const& send, _T *recv, int rank, int tag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Send and receive a single object. Deduce the data type.
/**
  \param send The object to send.
  \param dest Rank of destination (integer)
  \param sendTag Message tag (integer)
  \param recv The object to receive.
  \param source Rank of source (integer)
  \param recvTag Message tag (integer)
  \param comm Communicator (handle)
*/
template<typename _T>
void
sendRecv(_T const& send, int dest, int sendTag,
         _T *recv, int source, int recvTag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Sendrecv(). Send and receive a single object.
/**
  \param send The object to send.
  \param sendType %Data type of each send buffer element (handle)
  \param dest Rank of destination (integer)
  \param sendTag Message tag (integer)
  \param recv The object to receive.
  \param recvType %Data type of each receive buffer element (handle)
  \param source Rank of source (integer)
  \param recvTag Message tag (integer)
  \param comm Communicator (handle)
*/
template<typename _T>
void
sendRecv(_T const& send, MPI_Datatype sendType, int dest, int sendTag,
         _T *recv, MPI_Datatype recvType, int source, int recvTag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Sendrecv(). Use (exactly-sized) vectors.
/**
  \param send The vector to send.
  \param dest Rank of destination.
  \param recv The vector to receive.
  \param source Rank of source.
  \param tag Message tag.
  \param comm Communicator.
*/
template<typename _T>
void
sendRecv(std::vector<_T> const& send, int dest,
         std::vector<_T>* recv, int source, int tag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Sendrecv(). Use the same tag for the send and receive.
/**
  \param sendBuf Initial address of send buffer (choice)
  \param sendCount Number of elements in send buffer. Note that this is 
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param dest Rank of destination (integer)
  \param recvBuf Initial address of receive buffer (choice)
  \param recvCount Exact number of elements in receive buffer. Note that this is
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param source Rank of source (integer)
  \param tag Message tag (integer)
  \param comm Communicator (handle)
*/
template<typename _T>
void
sendRecv(_T const* sendBuf, std::size_t sendCount, int dest,
         _T *recvBuf, std::size_t recvCount, int source, int tag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Sendrecv(). Deduce the MPI data type.
/**
  \param sendBuf Initial address of send buffer (choice)
  \param sendCount Number of elements in send buffer. Note that this is 
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param dest Rank of destination (integer)
  \param sendTag Message tag (integer)
  \param recvBuf Initial address of receive buffer (choice)
  \param recvCount Exact number of elements in receive buffer. Note that this is
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param source Rank of source (integer)
  \param recvTag Message tag (integer)
  \param comm Communicator (handle)
*/
template<typename _T>
void
sendRecv(_T const* sendBuf, std::size_t sendCount, int dest, int sendTag,
         _T *recvBuf, std::size_t recvCount, int source, int recvTag,
         MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Sendrecv() that checks for overflow in the \c count argument.
/**
  \param sendBuf Initial address of send buffer (choice)
  \param sendCount Number of elements in send buffer. Note that this is 
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param sendType %Data type of each send buffer element (handle)
  \param dest Rank of destination (integer)
  \param sendTag Message tag (integer)
  \param recvBuf Initial address of receive buffer (choice)
  \param recvCount Exact number of elements in receive buffer. Note that this is
  specified using \c std::size_t. The conversion to \c int will be checked.
  \param recvType %Data type of each receive buffer element (handle)
  \param source Rank of source (integer)
  \param recvTag Message tag (integer)
  \param comm Communicator (handle)
*/
void
sendRecv(void const* sendBuf, std::size_t sendCount, MPI_Datatype sendType,
         int dest, int sendTag,
         void *recvBuf, std::size_t recvCount, MPI_Datatype recvType,
         int source, int recvTag,
         MPI_Comm comm = MPI_COMM_WORLD);

/** @} */ // End of mpiWrapperSendRecv group.



/** \defgroup mpiWrapperIsendIrecv MPI_Isend() and MPI_Irecv()
@{
*/

/// Wrapper for MPI_Isend().
/**
   \param buf The vector of elements to send.
   \param dest Rank of destination.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.
*/
template<typename _T>
MPI_Request
iSend(std::vector<_T> const& buf, int dest, int tag,
      MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Recv() that checks for overflow in the \c count argument.
/**
   \param buf The vector of elements to receive. The number of elements received
   must not exceed the buffer size.
   \param source Rank of source.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.
*/
template<typename _T>
MPI_Request
iRecv(std::vector<_T>* buf, int source, int tag,
      MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Isend().
/**
   \param buf Initial address of send buffer.
   \param count Number of elements in send buffer.
   \param dest Rank of destination.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.

   Deduce the MPI data type.
*/
template<typename _T>
MPI_Request
iSend(_T const* buf, std::size_t count, int dest, int tag,
      MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Recv() that checks for overflow in the \c count argument.
/**
   \param buf Initial address of receive buffer.
   \param count Maximum number of elements in receive buffer.
   \param source Rank of source.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.

   Deduce the MPI data type.
*/
template<typename _T>
MPI_Request
iRecv(_T* buf, std::size_t count, int source, int tag,
      MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Isend().
/**
   \param buf Initial address of send buffer.
   \param count Number of elements in send buffer. Note that this is specified
   using \c std::size_t. The conversion to \c int will be checked.
   \param datatype %Data type of each send buffer element.
   \param dest Rank of destination.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.

   \note The communicator is the last parameter so that it can have a 
   default value.
*/
MPI_Request
iSend(void const* buf, std::size_t count, MPI_Datatype datatype, int dest,
      int tag, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Recv() that checks for overflow in the \c count argument.
/**
   \param buf Initial address of receive buffer.
   \param count Maximum number of elements in receive buffer. Note that this is
   specified using \c std::size_t. The conversion to \c int will be checked.
   \param datatype %Data type of each receive buffer element.
   \param source Rank of source.
   \param tag Message tag.
   \param comm Communicator.
   \return The MPI request.

   \note The communicator is the last parameter so that it can have a 
   default value.
*/
MPI_Request
iRecv(void* buf, std::size_t count, MPI_Datatype datatype, int source,
      int tag, MPI_Comm comm = MPI_COMM_WORLD);

/** @} */ // End of mpiWrapperIsendIrecv group.




/** \defgroup mpiWrapperStatus MPI_Probe(), MPI_Wait(), and MPI_Get_count()
@{
*/

/// Wrapper for MPI_Probe().
/**
   \param source Source rank or MPI_ANY_SOURCE.
   \param tag Tag value or MPI_ANY_TAG.
   \param comm Communicator.
   \return The MPI status.
*/
MPI_Status
probe(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Probe().
/**
   \param source Source rank or MPI_ANY_SOURCE.
   \param tag Tag value or MPI_ANY_TAG.
   \param comm Communicator.
   \param status The MPI status.

   Call MPI_Probe() and check the return status.
*/
void
probe(int source, int tag, MPI_Comm comm, MPI_Status* status);

/// Wrapper for MPI_Wait().
/**
   \param request The MPI request.

   Ignore the status.
*/
void
wait(MPI_Request* request);

/// Wrapper for MPI_Wait().
/**
   \param request The MPI request.
   \param status The MPI status.
*/
void
wait(MPI_Request* request, MPI_Status* status);

/// Wrapper for MPI_Wait().
/**
\param request The MPI request.
\param count The exact number of elements that must be received.   

The MPI data type will be deduced.
*/
template<typename _T>
void
wait(MPI_Request* request, std::size_t count);

/// Wrapper for MPI_Waitall().
/**
   \param requests The vector of MPI requests.

   Ignore the statuses.
*/
void
waitAll(std::vector<MPI_Request>* requests);

/// Wrapper for MPI_Waitall().
/**
   \param requests The vector of MPI requests.
   \param statuses The vector of MPI statuses.
*/
void
waitAll(std::vector<MPI_Request>* requests, std::vector<MPI_Status>* statuses);

/// Wrapper for MPI_Waitall().
/**
   \param count The number of requests.
   \param requests The array of MPI requests.
   \param statuses The array of MPI statuses. May be MPI_STATUSES_IGNORE.

   This is a simple wrapper. Just check the return code.
*/
void
waitAll(int count, MPI_Request* requests, MPI_Status* statuses);

/// Wrapper for MPI_Get_count().
/**
\param status The MPI status.
\return The number of top-level elements.

The MPI data type will be deduced.
*/
template<typename _T>
int
getCount(MPI_Status const& status);

/// Wrapper for MPI_Get_count().
/**
\param status The MPI status.
\param datatype The MPI data type.
\return The number of top-level elements.
*/
int
getCount(MPI_Status const& status, MPI_Datatype datatype);

/** @} */ // End of mpiWrapperStatus group.



/** \defgroup mpiWrapperGather MPI_Gather()
    If you are gathering a data type that can be deduced with Data,
    use one of the following. (For the first the root process is the 
    one with rank zero.) Note that the output vector is automatically resized
    to the size of the communicator group.
    \code
    std::vector<std::size_t> pointSizes = mpi::gather(points.size(), comm);
    \endcode
    \code
    std::vector<std::size_t> pointSizes = mpi::gather(points.size(), comm, root);
    \endcode
    The more general interface is a simple wrapper for \c MPI_Gather().
    @{
*/

/// Wrapper for MPI_Gather().
/**
  \param input The object to send.
  \param comm Communicator.
  \param root Rank of the receiving process.
  \return The gathered objects. (Non-empty only at the root.)

  The MPI type is deduced using Data.
*/
template<typename _T>
std::vector<_T>
gather(_T const& input, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Gather(). Check the MPI version to use the appropriate signature.
/**
  \param sendbuf Starting address of send buffer (choice).
  \param sendcount Number of elements in send buffer (integer).
  \param sendtype %Data type of send buffer elements (handle)
  \param recvbuf Address of receive buffer (choice, significant only at root).
  \param recvcount Number of elements for any single receive (integer,
  significant only at root)
  \param recvtype %Data type of recv buffer elements (significant only at root)
  (handle).
  \param comm Communicator (handle).
  \param root Rank of receiving process (integer).
*/
void
gather(void const* sendbuf, std::size_t sendcount, MPI_Datatype sendtype,
       void* recvbuf, std::size_t recvcount, MPI_Datatype recvtype,
       MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Gatherv().
/**
  \param send The send vector.
  \param comm Communicator.
  \param root Rank of receiving process.
  \return The gathered objects. (Non-empty only at the root.)

  At the root, calculate the receive counts and displacements and allocate 
  memory in the receive vector before gathering the objects.
*/
template<typename _T>
std::vector<_T>
gather(std::vector<_T> const& send, MPI_Comm comm = MPI_COMM_WORLD,
       int root = 0);

/// Wrapper for MPI_Gatherv().
/**
  \param sendbuf Starting address of send buffer (choice).
  \param sendcount Number of elements in send buffer (integer).
  \param recvbuf Address of receive buffer (choice, significant only at root).
  \param recvcounts integer array (of length group size) containing the number
  of elements that are received from each process (significant only at root)
  \param displs integer array (of length group size). Entry i specifies the
  displacement relative to recvbuf at which to place the incoming data from 
  process i (significant only at root)
  \param comm Communicator (handle).
  \param root Rank of receiving process (integer).

  Deduce the MPI data type using Data.
*/
template<typename _T>
void
gather(_T const* sendbuf, int sendcount, _T* recvbuf, int const* recvcounts,
       int const* displs, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Gatherv().
/**
  \param sendbuf Starting address of send buffer (choice).
  \param sendcount Number of elements in send buffer (integer).
  \param sendtype %Data type of send buffer elements (handle)
  \param recvbuf Address of receive buffer (choice, significant only at root).
  \param recvcounts integer array (of length group size) containing the number
  of elements that are received from each process (significant only at root)
  \param displs integer array (of length group size). Entry i specifies the
  displacement relative to recvbuf at which to place the incoming data from 
  process i (significant only at root)
  \param recvtype %Data type of recv buffer elements (significant only at root)
  (handle).
  \param comm Communicator (handle).
  \param root Rank of receiving process (integer).

  Check the MPI version to use the appropriate signature.
*/
void
gather(void const* sendbuf, int sendcount, MPI_Datatype sendtype,
       void* recvbuf, int const* recvcounts, int const* displs,
       MPI_Datatype recvtype, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Gather().
/**
  \param input The object to send.
  \param comm Communicator.
  \return The gathered objects.

  The MPI type is deduced using Data.
*/
template<typename _T>
std::vector<_T>
allGather(_T const& input, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allgather().
/**
  \param sendbuf Starting address of send buffer.
  \param sendcount Number of elements in send buffer.
  \param sendtype %Data type of send buffer elements
  \param recvbuf Address of receive buffer.
  \param recvcount Number of elements received from an process.
  \param recvtype %Data type of recv buffer elements.
  \param comm Communicator.
*/
void
allGather(void const* sendbuf, std::size_t sendcount, MPI_Datatype sendtype,
          void* recvbuf, std::size_t recvcount, MPI_Datatype recvtype,
          MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allgatherv().
/**
  \param sendbuf The vector of objects to send.
  \param comm Communicator.
  \return The gathered objects.
*/
template<typename _T>
std::vector<_T>
allGather(std::vector<_T> const& sendBuf, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allgatherv().
/**
  \param send The vector of objects to send.
  \param comm Communicator.
  \return The packed array of arrays of the objects.
*/
template<typename _T>
container::PackedArrayOfArrays<_T>
allGatherPacked(std::vector<_T> const& send, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allgatherv().
/**
  \param sendbuf Starting address of send buffer.
  \param sendcount Number of elements in send buffer.
  \param sendtype %Data type of send buffer elements
  \param recvbuf Address of receive buffer.
  \param recvcounts Integer array (of length group size) containing the number
  of elements that are to be received from each process.
  \param displs Integer array (of length group size). Entry i specifies the
  displacement (relative to recvbuf ) at which to place the incoming data
  from process i.
  \param recvtype %Data type of recv buffer elements.
  \param comm Communicator.
*/
void
allGather(void const* sendbuf, int sendcount, MPI_Datatype sendtype,
          void* recvbuf, int const* recvcounts, int const* displs,
          MPI_Datatype recvtype, MPI_Comm comm = MPI_COMM_WORLD);

/** @} */ // End of mpiWrapperGather group.



/** \defgroup mpiWrapperReduce MPI_Reduce()
    For single, built-in types, the wrapper returns the reduced value.
    \code
    std::size_t const total = mpi::reduce(points.size(), MPI_SUM, comm);
    \endcode
    The result is only valid at the root. If the root is not process zero,
    you will need to specify it.
    \code
    std::size_t const total = mpi::reduce(points.size(), MPI_SUM, comm, root);
    \endcode
    For a \c std::vector of values, pass the vectors as arguments.
    \code
    std::vector<std::size_t> sizes;
    ...
    std::vector<std::size_t> totals(sizes.size());
    mpi::reduce(sizes, &totals, MPI_SUM, comm);
    \endcode
    The more general interface is a simple wrapper for \c MPI_Reduce().
    @{
*/

/// Wrapper for MPI_Reduce(). Reduction for a single object.
/**
  \param object The object to send in the reduction.
  \param op Operation (handle).
  \param comm Communicator (handle).
  \param root The rank of the root process.

  The result is only valid at the root.
  \note \c _T must be a built-in MPI type.
*/
template<typename _T>
_T
reduce(_T object, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Reduce().
/**
  \param sendObjects The vector of elements to send.
  \param recvObjects The vector of reduced objects to receive.
  \param op Operation (handle).
  \param comm Communicator (handle).
  \param root The rank of the root process.

  Deduce the data type. The result is only valid at the root. There, the
  output vector will be resized before being filled.
*/
template<typename _T>
void
reduce(std::vector<_T> const& sendObjects, std::vector<_T>* recvObjects,
       MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Reduce().
/**
  \param sendbuf Starting address of send buffer (choice).
  \param recvbuf Starting address of receive buffer (choice).
  \param count Number of elements in send buffer (integer). Check for overflow.
  \param datatype %Data type of elements of send buffer (handle).
  \param op Operation (handle).
  \param comm Communicator (handle).
  \param root The rank of the root process.

  Check the MPI version to use the appropriate signature.
*/
void
reduce(void const* sendbuf, void* recvbuf, std::size_t count,
       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD,
       int root = 0);

/** @} */ // End of mpiWrapperAllreduce group.



/** \defgroup mpiWrapperAllreduce MPI_Allreduce()
    For single, built-in types, the wrapper returns the reduced value.
    \code
    std::size_t const total = mpi::allReduce(points.size(), MPI_SUM, comm);
    \endcode
    For a \c std::vector of values, pass the vectors as arguments.
    \code
    std::vector<std::size_t> sizes;
    ...
    std::vector<std::size_t> totals(sizes.size());
    mpi::allReduce(sizes, &totals, MPI_SUM, comm);
    \endcode
    The more general interface is a simple wrapper for \c MPI_Allreduce().
    @{
*/

/// Wrapper for MPI_Allreduce(). Reduction for a single object.
/**
  \param object The object to send in the reduction.
  \param op Operation (handle).
  \param comm Communicator (handle).

  \note \c _T must be a built-in MPI type.
*/
template<typename _T>
_T
allReduce(_T object, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allreduce() using \c MPI_SUM. Reduction for a single object.
/**
  \param object The object to send in the reduction.
  \param comm Communicator.

  Check for overflow with certain types.
  \note \c _T must be a built-in MPI type.
*/
template<typename _T>
_T
allReduceSum(_T object, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allreduce().
/**
  \param sendObjects The vector of elements to send.
  \param recvObjects The vector of reduced objects to receive.
  \param op Operation (handle).
  \param comm Communicator (handle).

  Check the sizes of the vectors. Deduce the data type.
*/
template<typename _T>
void
allReduce(std::vector<_T> const& sendObjects, std::vector<_T>* recvObjects,
          MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD);

/// Wrapper for MPI_Allreduce().
/**
  \param sendbuf Starting address of send buffer (choice).
  \param recvbuf Starting address of receive buffer (choice).
  \param count Number of elements in send buffer (integer). Check for overflow.
  \param datatype %Data type of elements of send buffer (handle).
  \param op Operation (handle).
  \param comm Communicator (handle).

  Check the MPI version to use the appropriate signature.
*/
void
allReduce(void const* sendbuf, void* recvbuf, std::size_t count,
          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD);

/** @} */ // End of mpiWrapperAllreduce group.



/** \defgroup mpiWrapperBcast MPI_Bcast()
    For each of the bcast() functions, the root is an optional argument.
    You only need to specify it if its value is non-zero.
    To broadcast a single object whose type can be deduced with Data, 
    use the following interface.
    \code
    double value;
    if (mpi::commRank(comm) == 0) {
      value = ...;
    }
    mpi::bcast(&value, comm);
    \endcode
    
    To broadcast a \c std::vector of objects whose type can be deduced with
    Data, use the following interface.
    \code
    std::vector<double> values;
    if (mpi::commRank(comm) == 0) {
      // Set the values.
      ...
    }
    mpi::bcast(&values, comm);
    \endcode
    Note that on non-root processes, the vector of values is resized.
    
    If the vectors don't need to be resized, use bcastNoResize().
    \code
    std::vector<double> values(size);
    if (mpi::commRank(comm) == 0) {
      // Set the values.
      ...
    }
    mpi::bcastNoResize(&values, comm);
    \endcode

    If you aren't storing the objects in a \c std::vector,
    there is an interface for broadcasting a buffer of objects whose type
    can be deduced.
    \code
    double values[size];
    if (mpi::commRank(comm) == 0) {
      // Set the values.
      ...
    }
    mpi::bcast(values, size, comm);
    \endcode
    If the data type cannot be deduced, then you need to specify it.
    \code
    mpi::bcast(values, size, datatype, comm);
    \endcode

    @{
*/

/// Wrapper for MPI_Bcast(). Broadcast a single object.
/**
  \param object The object to broadcast.
  \param comm Communicator.
  \param root Rank of broadcast root (integer).
*/
template<typename _T>
void
bcast(_T* object, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Bcast(). Broadcast a vector of objects.
/**
  \param objects The objects to broadcast.
  \param comm Communicator.
  \param root Rank of broadcast root (integer).

  At non-root processes, the vector will be resized.
*/
template<typename _T>
void
bcast(std::vector<_T>* objects, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/// Wrapper for MPI_Bcast(). Broadcast a vector of objects.
/**
  \param objects The objects to broadcast.
  \param comm Communicator.
  \param root Rank of broadcast root (integer).

  \pre The \c objects vector must have the same size on all processes in the 
  communicator.
*/
template<typename _T>
void
bcastNoResize(std::vector<_T>* objects, MPI_Comm comm = MPI_COMM_WORLD,
              int root = 0);

/// Wrapper for MPI_Bcast(). The data type is deduced.
/**
  \param buffer Starting address of buffer (choice).
  \param count Number of entries in buffer (integer).
  \param comm Communicator.
  \param root Rank of broadcast root (integer).
*/
template<typename _T>
void
bcast(_T* buffer, std::size_t count, MPI_Comm comm = MPI_COMM_WORLD,
      int root = 0);

/// Wrapper for MPI_Bcast().
/**
  \param buffer Starting address of buffer (choice).
  \param count Number of entries in buffer (integer).
  \param datatype %Data type of buffer (handle).
  \param comm Communicator.
  \param root Rank of broadcast root (integer).
*/
void
bcast(void* buffer, std::size_t count, MPI_Datatype datatype,
      MPI_Comm comm = MPI_COMM_WORLD, int root = 0);

/** @} */ // End of mpiWrapperBcast group.



/** \defgroup mpiWrapperScatter MPI_Scatter()
 */


/// Wrapper for MPI_Scatter().
/**
   \param send Vector of values to send (significant only at root).
   \param comm Communicator.
   \param root Rank of sending process.

   Send a single value to each process.
   \return The scattered value.
*/
template<typename _T>
inline
_T
scatter(std::vector<_T> const& send, MPI_Comm comm = MPI_COMM_WORLD,
        int root = 0);

/// Wrapper for MPI_Scatter().
/**
   \param sendbuf Address of send buffer (significant only at root).
   \param count Number of elements sent to each process This is also the
   number of elements in receive buffer.
   \param recvbuf Address of receive buffer.
   \param comm Communicator.
   \param root Rank of sending process.
*/
template<typename _T>
inline
void
scatter(_T const* sendbuf, std::size_t const count, _T* recvbuf,
        MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/// Wrapper for MPI_Scatter().
/**
   \param sendbuf Address of send buffer (significant only at root).
   \param sendcount Number of elements sent to each process (significant only
   at root).
   \param sendtype Datatype of send buffer elements (significant only at root).
   \param recvbuf Address of receive buffer.
   \param recvcount Number of elements in receive buffer.
   \param recvtype Datatype of receive buffer elements.
   \param comm Communicator.
   \param root Rank of sending process.
*/
void
scatter(void const* sendbuf, std::size_t sendcount, MPI_Datatype sendtype,
        void* recvbuf, std::size_t recvcount, MPI_Datatype recvtype,
        MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/** @} */ // End of mpiWrapperScatter group.


/** @} */ // End of mpiWrapper group.


} // namespace mpi
}

#define __mpi_wrapper_tcc__
#include "stlib/mpi/wrapper.tcc"
#undef __mpi_wrapper_tcc__

#endif

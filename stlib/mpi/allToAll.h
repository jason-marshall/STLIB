// -*- C++ -*-

/**
  \file mpi/allToAll.h
  \brief Convenience functions for MPI_AllToAll().
*/

#if !defined(__mpi_allToAll_h__)
#define __mpi_allToAll_h__

#include "stlib/mpi/wrapper.h"

#include "stlib/container/PackedArrayOfArrays.h"

namespace stlib
{
namespace mpi
{

/** \addtogroup mpiWrapper
    @{
*/



/** \defgroup mpiWrapperAllToAll MPI_Alltoall()
    These functions are used in the all-to-all communication pattern. Suppose 
    that you store the objects to send in a container::PackedArrayOfArrays,
    where the objects are grouped according to the process to which they will 
    be sent. If the data type can be deduced using Data, then use the 
    simple interface. An example is shown below.
    \code
    container::PackedArrayOfArrays<std::size_t> send;
    // Build the array of arrays to send.
    ...
    container::PackedArrayOfArrays<std::size_t> receive;
    allToAll(send, &receive, comm);
    \endcode
    Note that the \c receive container will be automatically resized and
    then filled. If the data type cannot be deduced, you will need to 
    specify it.
    \code
    allToAll(send, &receive, datatype, comm);
    \endcode

    If you don't use the information about from where the objects came, you
    can use the interface that collects the received objects in a
    \c std::vector. Below is an example.
    \code
    std::vector<double> sendObjects;
    std::vector<std::size_t> sendCounts;
    // Build the data to send.
    ...
    std::vector<double> recvBuf;
    allToAll(sendObjects, sendCounts, &recvBuf, comm);
    \endcode

    There are allToAll() functions that provide simple wrappers for 
    \c MPI_Alltoall() and MPI_Alltoallv(). These just perform error checking.

    @{
*/


/// Perform the all-to-all communication.
/**
  The MPI data type will be deduced using \c Data<_T>::type().
*/
template<typename _T>
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         container::PackedArrayOfArrays<_T>* receive,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Perform the all-to-all communication.
/**
  The MPI data type will be deduced using \c Data<_T>::type().
*/
template<typename _T>
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         std::vector<_T>* receive, MPI_Comm comm = MPI_COMM_WORLD);


/// Perform the all-to-all communication.
/**
  \param sendBuf Contains the objects that will be sent to each process.
  \param sendCounts_ The number of objects to send to each process. This
  allows one to interpret sendBuf as a packed array of arrays.
  \param recvBuf Receives the objects from each process. This array will
  be resized before being filled with values.
  \param comm The MPI intra-communicator.
*/
template<typename _T>
void
allToAll(std::vector<_T> const& sendBuf,
         std::vector<std::size_t> const& sendCounts_,
         std::vector<_T>* recvBuf, MPI_Comm comm = MPI_COMM_WORLD);


/// Wrapper for MPI_Alltoall(). Send a single object to each process.
/**
   \param send The objects to send.
   \param receive The received objects. The vector must be the correct size.
   \param comm Communicator.

   Deduce the MPI data type.
*/
template<typename _T>
void
allToAll(std::vector<_T> const& send, std::vector<_T>* receive,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Wrapper for MPI_Alltoall()
/**
   \param sendbuf Starting address of send buffer (choice).
   \param sendcount Number of elements to send to each process (integer).
   \param sendtype Data type of send buffer elements (handle).
   \param recvbuf Address of receive buffer (choice).
   \param recvcount Number of elements received from any process (integer).
   \param recvtype Data type of receive buffer elements (handle).
   \param comm Communicator (handle).

   Check the MPI version and perform conversions from pointers to const to 
   pointers for MPI 2. Check the error code.
*/
void
allToAll(void const* sendbuf, int sendcount, MPI_Datatype sendtype,
         void* recvbuf, int recvcount, MPI_Datatype recvtype,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Wrapper for MPI_Alltoallv()
/**
   \param send Starting address of send buffer.
   \param sendCounts Integer array equal to the group size specifying the number
   of elements to send to each processor.
   \param recv Address of receive buffer.
   \param recvCounts Integer array equal to the group size specifying the 
   maximum number of elements that can be received from each processor.
   \param comm Communicator.

   Use Data to deduce the MPI type.
*/
template<typename _T>
void
allToAll(_T const* send, int const* sendCounts, _T* recv, int const* recvCounts,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Wrapper for MPI_Alltoallv()
/**
   \param sendbuf Starting address of send buffer (choice).
   \param sendcounts Integer array equal to the group size specifying the number
   of elements to send to each processor.
   \param sendtype Data type of send buffer elements (handle).
   \param recvbuf Address of receive buffer (choice).
   \param recvcounts Integer array equal to the group size specifying the 
   maximum number of elements that can be received from each processor.
   \param recvtype Data type of receive buffer elements (handle).
   \param comm Communicator (handle).

   Deduce the send and receive displacements from the counts.
*/
void
allToAll(void const* sendbuf, int const* sendcounts, MPI_Datatype sendtype,
         void *recvbuf, int const* recvcounts, MPI_Datatype recvtype,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Wrapper for MPI_Alltoallv()
/**
   \param sendbuf Starting address of send buffer (choice).
   \param sendcounts Integer array equal to the group size specifying the number
   of elements to send to each processor.
   \param sdispls Integer array (of length group size). Entry j specifies the
   displacement (relative to sendbuf from which to take the outgoing
   data destined for process j.
   \param sendtype Data type of send buffer elements (handle).
   \param recvbuf Address of receive buffer (choice).
   \param recvcounts Integer array equal to the group size specifying the 
   maximum number of elements that can be received from each processor.
   \param rdispls Integer array (of length group size). Entry i specifies the
   displacement (relative to recvbuf at which to place the incoming data
   from process i.
   \param recvtype Data type of receive buffer elements (handle).
   \param comm Communicator (handle).

   Check the MPI version and perform conversions from pointers to const to 
   pointers for MPI 2. Check the error code.
*/
void
allToAll(void const* sendbuf, int const* sendcounts,
         int const* sdispls, MPI_Datatype sendtype, void *recvbuf,
         int const* recvcounts, int const* rdispls, MPI_Datatype recvtype,
         MPI_Comm comm = MPI_COMM_WORLD);


/// Initiate the non-blocking, all-to-all communication.
/**
  The MPI data type will be deduced using \c Data<_T>::type(). Use waitAll()
  to wait for the communications to complete.
*/
template<typename _T>
void
allToAll(container::PackedArrayOfArrays<_T> const& send,
         std::vector<_T>* receive,
         int tag,
         std::vector<MPI_Request>* sendRequests,
         std::vector<MPI_Request>* receiveRequests,
         MPI_Comm comm = MPI_COMM_WORLD);


/** @} */ // End of mpiWrapperAllToAll group.


/** @} */ // End of mpiWrapper group.


} // namespace mpi
}

#define __mpi_allToAll_tcc__
#include "stlib/mpi/allToAll.tcc"
#undef __mpi_allToAll_tcc__

#endif

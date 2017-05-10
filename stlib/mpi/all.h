// -*- C++ -*-

/*!
  \file mpi/all.h
  \brief Convenience header that include all headers in the mpi package.
*/

#if !defined(__mpi_all_h__)
#define __mpi_all_h__

#include "stlib/mpi/Data.h"
#include "stlib/mpi/allToAll.h"
#include "stlib/mpi/statistics.h"
#include "stlib/mpi/wrapper.h"

namespace stlib
{
//! Some functions that extend the capabilities of the MPI library are defined in this namespace.
namespace mpi
{
}
}

/*!
\mainpage MPI Wrapper and Utilities

\par Wrappers.
MPI provides very useful functionality, but has a terrible interface. Because 
it provides interoperability with Fortran, if you program in C++ (or C) you 
are effectively shackled to a corpse. This package does not aim to fix MPI.
Rather, it is a modest effort that eases some tasks. There are 
\ref mpiWrapper. These provide a more convienient interface for select 
functions, specifically, the functions that I often use. Unlike MPI functions,
these do not return error codes. Error codes are checked internally, and 
exceptions are thrown on errors. MPI is not 64-bit compliant. The functions
help to identify runtime errors that may occur as a result.
There are also functions for computing \ref mpiStatistics "statistics" for 
distributed values. 


\par Data types.
The mpi::Data class maps tyes to MPI types. This is particularly useful in 
template programming. The class is specialized for the built-in numeric types.
You can add specializations for your own data structures. (There is already
a specialization for \c std::array.) Use the \c type() member function to 
get the MPI data type for a C++ type. For built-in types, this member is
static. Thus, you can use it as shown below.
\code
mpi::send(buf, count, mpi::Data<double>::type(), dest, tag, comm);
\endcode
Here, \c mpi::Data<double>::type() return \c MPI_DOUBLE. As mentioned above,
this feature is more useful in templated code.

\par
The general implementation of mpi::Data handles any class that is trivially
copyable. The \c type() member function is not static, so one must construct
the class. Below is an example usage.
\code
mpi::Data<Foo> data;
mpi::send(buf, count, data.type(), dest, tag, comm);
\endcode

\par
Using mpi::Data mitigates problems stemming from the fact that MPI is not
64-bit compliant. Namely, it can reduce the count argument 
to delay problems with overflow in the signed \c int parameter. That is,
sending a buffer using an MPI data type will have a smaller count than
sending it using the MPI_BYTE type. For example, using the type for
\c mpi::Data<std::array<double,3>> will result in a count that is a factor
of 24 smaller than the byte count of the buffer.

\par Notes on using MPI.
When using OpenMP, you can tell if it is enabled with by checking if
the macro \c _OPENMP is defined. With MPI, the situation is different.
While the macro \c MPI_VERSION is defined in MPI 3 and later, it is
defined in the header file mpi.h.  Thus, the order of including
headers is important. If you have headers that check \c MPI_VERSION
and are building an MPI application, then you need to include mpi.h
before the other headers.
*/

#endif

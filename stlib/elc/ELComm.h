// -*- C++ -*-

/*!
  \file ELComm.h
  \brief Base class for Eulerian and Lagrangian communicators.
*/

#if !defined(__elc_ELComm_h__)
#define __elc_ELComm_h__

#include "stlib/ext/array.h"
#include "stlib/geom/kernel/BBox.h"

#include <vector>

#include <mpi.h>

namespace stlib
{
namespace elc
{

//! The two ways of representing a vertex in a connectivity array.
enum VertexIdentifierStyle {LocalIndices, GlobalIdentifiers};

//! Base class for Eulerian and Lagrangian communicators.
/*!
  \param N is the space dimension.  1, 2 and 3 are supported.
  \param T is the floating point number type.

  This is used to define common types and tags for communication.
*/
template<std::size_t N, typename T>
class ELComm
{
  //
  // Protected types.
  //

protected:

  //! The number type.
  typedef T Number;
  //! A Cartesian point.
  typedef std::array<Number, N> Point;
  //! An indexed face.
  typedef std::array<int, N> IndexedFace;
  //! A bounding box.
  typedef geom::BBox<N, Number> BBox;
#ifdef ELC_USE_CPP_INTERFACE
  //! An MPI request.
  typedef MPI::Request MpiRequest;
  //! Status for an MPI request.
  typedef MPI::Status MpiStatus;
#else
  //! An MPI request.
  typedef MPI_Request MpiRequest;
  //! Status for an MPI request.
  typedef MPI_Status MpiStatus;
#endif

  //
  // Enumerations.
  //

protected:

  //! Tags for the different communications.
  enum {TagIdentifiers,
        TagPositions,
        TagVelocities,
        TagPressures,
        TagFaceData
       };

  //
  // Member data.
  //

protected:

  //! The group that includes both the Eulerian and Lagrangian processors.
#ifdef ELC_USE_CPP_INTERFACE
  MPI::Intracomm _comm;
#else
  MPI_Comm _comm;
#endif

#ifdef ELC_USE_CPP_INTERFACE
  //! The MPI number type.
  MPI::Datatype _mpiNumber;
#else
  //! The MPI number type.
  MPI_Datatype _mpiNumber;
#endif

  //! The vertex identifier style.
  VertexIdentifierStyle _vertexIdentifierStyle;

  //
  // Not implemented.
  //

private:

  // Default constructor not implemented.
  ELComm();

  // Copy constructor not implemented.
  ELComm(const ELComm&);

  // Assignment operator not implemented.
  ELComm&
  operator=(const ELComm&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Constructor.
#ifdef ELC_USE_CPP_INTERFACE
  ELComm(const MPI::Intracomm& comm,
         VertexIdentifierStyle vertexIdentifierStyle) :
    _comm(comm.Dup()),
    _vertexIdentifierStyle(vertexIdentifierStyle)
  {
    if (sizeof(Number) == sizeof(float)) {
      _mpiNumber = MPI::FLOAT;
    }
    else if (sizeof(Number) == sizeof(double)) {
      _mpiNumber = MPI::DOUBLE;
    }
    else if (sizeof(Number) == sizeof(long double)) {
      _mpiNumber = MPI::LONG_DOUBLE;
    }
    else {
      assert(false);
    }
  }
#else
  ELComm(const MPI_Comm comm,
         VertexIdentifierStyle vertexIdentifierStyle) :
    _comm(),
    _vertexIdentifierStyle(vertexIdentifierStyle)
  {
    MPI_Comm_dup(comm, &_comm);

    if (sizeof(Number) == sizeof(float)) {
      _mpiNumber = MPI_FLOAT;
    }
    else if (sizeof(Number) == sizeof(double)) {
      _mpiNumber = MPI_DOUBLE;
    }
    else if (sizeof(Number) == sizeof(long double)) {
      _mpiNumber = MPI_LONG_DOUBLE;
    }
    else {
      assert(false);
    }
  }
#endif

  //! Destructor.
  virtual
  ~ELComm()
  {
#ifdef ELC_USE_CPP_INTERFACE
    _comm.Barrier();
    _comm.Free();
#else
    MPI_Barrier(_comm);
    MPI_Comm_free(&_comm);
#endif
  }

  // @}
};

} // namespace elc
}

#endif

// -*- C++ -*-

/*!
  \file LagrangianComm.h
  \brief Lagrangian communicator for the Eulerian-Lagrangian coupling.
*/

#if !defined(__elc_LagrangianComm_h__)
#define __elc_LagrangianComm_h__

#include "stlib/elc/ELComm.h"

#if defined(ELC_USE_CPP_INTERFACE) && !defined(PT2PT_BBOX_USE_CPP_INTERFACE)
#define PT2PT_BBOX_USE_CPP_INTERFACE
#endif
#include "stlib/concurrent/pt2pt_bbox.h"

#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/container/ArrayRef.h"

namespace stlib
{
namespace elc
{

//! Base class Lagrangian communicator for the Eulerian-Lagrangian coupling.
/*!
  \param N is the space dimension.  1, 2 and 3 are supported.
  \param T is the floating point number type.

  Implements the common functionality for boundaries and shells.
*/
template<std::size_t N, typename T>
class LagrangianComm :
  public ELComm<N, T>
{
  //
  // Private types.
  //

private:

  typedef ELComm<N, T> Base;

  //
  // Protected types.
  //

protected:

  //! A Cartesian point.
  typedef typename Base::Point Point;
  //! An indexed face.
  typedef typename Base::IndexedFace IndexedFace;
  //! A bounding box.
  typedef typename Base::BBox BBox;
  //! An MPI request.
  typedef typename Base::MpiRequest MpiRequest;
  //! Status for an MPI request.
  typedef typename Base::MpiStatus MpiStatus;

  //
  // Public types.
  //

public:

  //! The number type.
  typedef typename Base::Number Number;

  //
  // Using base members.
  //

private:

  //! The joint Eulerian/Lagrangian communicator.
  using Base::_comm;
  //! The MPI number type.
  using Base::_mpiNumber;
  //! The vertex identifier style.
  using Base::_vertexIdentifierStyle;

  //! The communication tag for node identifiers.
  using Base::TagIdentifiers;
  //! The communication tag for node positions.
  using Base::TagPositions;
  //! The communication tag for node velocities.
  using Base::TagVelocities;
  //! The communication tag for face data.
  using Base::TagFaceData;
  //! The communication tag for pressures.
  using Base::TagPressures;

  //
  // Member data.
  //

private:

#ifdef ELC_USE_CPP_INTERFACE
  //! The Lagrangian communicator.
  MPI::Intracomm _lagrangianCommunicator;
#else
  //! The Lagrangian communicator.
  MPI_Comm _lagrangianCommunicator;
#endif

  //! The Eulerian root.
  int _eulerianRoot;

  //! The set of Eulerian processors with which to communicate.
  std::vector<int> _eulerianProcessors;

  // Class for computing the point-to-point communication scheme.
  concurrent::PtToPt2Grp1Dom<N, T, std::array<int, 3>, int>
  _pointToPoint;

  std::vector<MpiRequest> _identifierRequests;
  std::vector<MpiRequest> _positionRequests;
  std::vector<MpiRequest> _velocityRequests;
  std::vector<MpiRequest> _connectivityRequests;

#if 0
  // CONTINUE REMOVE
  int _numNodes;
  int _numFaces;
#endif

  //! The pressures to be received from Eulerian processors.
  std::vector<std::vector<Number> > _pressures;
  container::ArrayRef<Number> _compositePressures;

  std::vector<MpiRequest> _pressureRequests;

  int _numPoints;

  //
  // Not implemented.
  //

private:

  // Default constructor not implemented.
  LagrangianComm();

  // Copy constructor not implemented.
  LagrangianComm(const LagrangianComm&);

  // Assignment operator not implemented.
  LagrangianComm&
  operator=(const LagrangianComm&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

#ifdef ELC_USE_CPP_INTERFACE
  //! Construct from the communicators and Eulerian information.
  /*!
    \param comm is the communicator that contains the Eulerian and
    Lagrangian processors.
    \param lagrangian is the Lagrangian communicator.  It is duplicated to
    avoid message conflicts.
    \param eulerianSize is the number of Eulerian processors.
    \param eulerianRoot is the rank of the Eulerian root in \c comm.
    \param vertexIdentifierStyle is either LocalIndices or GlobalIdentifiers.
  */
  LagrangianComm(const MPI::Comm& comm, const MPI::Intracomm& lagrangian,
                 const int eulerianSize, const int eulerianRoot,
                 VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, vertexIdentifierStyle),
    _lagrangianCommunicator(lagrangian.Dup()),
    _eulerianRoot(eulerianRoot),
    _pointToPoint(comm, lagrangian, eulerianSize, eulerianRoot),
    _identifierRequests(),
    _positionRequests(),
    _velocityRequests(),
    _connectivityRequests(),
    _pressures(),
    _compositePressures(0, 0),
    _pressureRequests(),
    _numPoints() {}
#else
  //! Construct from the communicators and Eulerian information.
  /*!
    \param comm is the communicator that contains the Eulerian and
    Lagrangian processors.
    \param lagrangian is the Lagrangian communicator.  It is duplicated to
    avoid message conflicts.
    \param eulerianSize is the number of Eulerian processors.
    \param eulerianRoot is the rank of the Eulerian root in \c comm.
    \param vertexIdentifierStyle is either LocalIndices or GlobalIdentifiers.
  */
  LagrangianComm(const MPI_Comm comm, const MPI_Comm lagrangian,
                 const int eulerianSize, const int eulerianRoot,
                 VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, vertexIdentifierStyle),
    _lagrangianCommunicator(),
    _eulerianRoot(eulerianRoot),
    _pointToPoint(comm, lagrangian, eulerianSize, eulerianRoot),
    _identifierRequests(),
    _positionRequests(),
    _velocityRequests(),
    _connectivityRequests(),
    _pressures(),
    _compositePressures(0, 0),
    _pressureRequests(),
    _numPoints()
  {
    MPI_Comm_dup(lagrangian, &_lagrangianCommunicator);
  }
#endif

  //! Destructor.  Free the communicator.
  virtual
  ~LagrangianComm()
  {
#ifdef ELC_USE_CPP_INTERFACE
    _lagrangianCommunicator.Free();
#else
    MPI_Comm_free(&_lagrangianCommunicator);
#endif
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  // Disabled documentation for the identifiers parameter:
  // If the solid solver does not have this data, it should pass 0.
  // Then identifiers will be computed on the Lagrangian nodes.

  //! Post sends for sending the local Lagrangian mesh to the relevant Eulerian processors.
  /*!
    \param numNodes is the number of nodes in the mesh.
    \param identifiers are the global mesh node identifiers.
    \param positions are the mesh node positions.  This is an array of N-tuples
    of numbers.  Each N-tuple is a Cartesian point.
    \param velocities are the mesh node velocities.
    \param numFaces is the number of faces in the mesh.
    \param connectivities describes the connectivites of the
    nodes to form faces.  This is an array of N-tuples of integers.  Each
    N-tuple represents on indexed face (line segment in 2-D, triangle in 3-D).
    If the vertex identifier style is \c LocalIndices, then each index is in
    the range [0..numFaces).  If the style is \c GlobalIdentifers, then each
    element of \c connectivities is listed in the \c identifiers array.
  */
  void
  sendMesh(std::size_t numNodes, const int* identifiers,
           const Number* positions, const Number* velocities,
           std::size_t numFaces, const int* connectivities);

  //! Wait for the sends to complete.
  void
  waitForMesh();

  //! Post receives for the pressure from the relevant Eulerian processors.
  void
  receivePressure(std::size_t numPoints, Number* pressures);

  //! Wait for the pressure receives to complete.  Composite the pressures.
  void
  waitForPressure();

  // @}
};


} // namespace elc
}

#define __elc_LagrangianComm_ipp__
#include "stlib/elc/LagrangianComm.ipp"
#undef __elc_LagrangianComm_ipp__

#endif

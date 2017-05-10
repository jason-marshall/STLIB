// -*- C++ -*-

/*!
  \file EulerianCommBoundary.h
  \brief Eulerian communicator for the Eulerian-Lagrangian coupling.
*/

#if !defined(__elc_EulerianCommBoundary_h__)
#define __elc_EulerianCommBoundary_h__

#include "stlib/elc/EulerianComm.h"

namespace stlib
{
namespace elc
{

//! Eulerian communicator for the Eulerian-Lagrangian coupling.
/*!
  \param N is the space dimension.  2 and 3 are supported.
  \param T is the floating point number type.  By default it is double.

  This class is instantiated in each Eulerian (fluid) processor.  The
  relevant portions of the boundary of the solid are received from the
  solid processors with either
  receiveMesh(const BBox& domain)
  or
  receiveMesh(const Number* domain)
  followed by a call to
  waitForMesh().

  One can access the node positions with either getPositions() or
  getPositionsData(); the node velocities can be access with getVelocities() or
  getVelocitiesData().  This processor must set the pressure at all node
  positions that lie within its domain.  The pressures may be manipulated
  through getPressures() or getPressuresData().

  After the pressures have been set, they are sent to the relevant Lagrangian
  processors with sendPressure() and waitForPressure().
*/
template < std::size_t N, typename T = double >
class EulerianCommBoundary :
  public EulerianComm<N, T>
{
  //
  // Private types.
  //

private:

  typedef EulerianComm<N, T> Base;
  typedef typename Base::Point Point;
  typedef typename Base::BBox BBox;
  typedef typename Base::MpiRequest MpiRequest;
  typedef typename Base::MpiStatus MpiStatus;
  typedef typename Base::IndexedFace IndexedFace;

  //
  // Public types.
  //

public:

  //! The number type.
  typedef T Number;

  //
  // Using base members.
  //

private:

  //! The joint Eulerian/Lagrangian communicator.
  using Base::_comm;
  //! The MPI number type.
  using Base::_mpiNumber;
  //! The communication tag for pressures.
  using Base::TagPressures;

  //! The node identifiers from Lagrangian processors.
  using Base::_identifiers;
  //! The node pressures to be sent to Lagrangian processors.
  using Base::_pressures;
  //! The mapping from node identifiers to node indices in the assembled boundary.
  using Base::_identifierToIndex;
  //! The assembled positions.
  using Base::_assembledPositions;
  //! The assembled pressures.
  using Base::_assembledPressures;
  //! Data from the Lagrangian processors with which we communicate.
  using Base::_lagrangianData;

  //
  // Member data.
  //

private:

  // MPI requests for the pressures sends.
  std::vector<MpiRequest> _pressureRequests;

  //
  // Not implemented.
  //

private:

  // Default constructor not implemented.
  EulerianCommBoundary();

  // Copy constructor not implemented.
  EulerianCommBoundary(const EulerianCommBoundary&);

  // Assignment operator not implemented.
  EulerianCommBoundary&
  operator=(const EulerianCommBoundary&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

#ifdef ELC_USE_CPP_INTERFACE
  //! Construct from the communicators and Lagrangian info.
  /*!
    \param comm is the communicator that contains the Eulerian and
    Lagrangian processors.
    \param eulerian is the Eulerian communicator.  It is duplicated to avoid
    message conflicts.
    \param lagrangianSize is the number of Lagrangian processors.
    \param lagrangianRoot is the rank of the Lagrangian root in \c comm.
    \param vertexIdentifierStyle is either LocalIndices or GlobalIdentifiers.
  */
  EulerianCommBoundary(const MPI::Comm& comm, const MPI::Intracomm& eulerian,
                       const int lagrangianSize, const int lagrangianRoot,
                       VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, eulerian, lagrangianSize, lagrangianRoot, vertexIdentifierStyle) {}
#else
  //! Construct from the communicators and Lagrangian info.
  /*!
    \param comm is the communicator that contains the Eulerian and
    Lagrangian processors.
    \param eulerian is the Eulerian communicator.  It is duplicated to avoid
    message conflicts.
    \param lagrangianSize is the number of Lagrangian processors.
    \param lagrangianRoot is the rank of the Lagrangian root in \c comm.
    \param vertexIdentifierStyle is either LocalIndices or GlobalIdentifiers.
  */
  EulerianCommBoundary(const MPI_Comm comm, const MPI_Comm eulerian,
                       const int lagrangianSize, const int lagrangianRoot,
                       VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, eulerian, lagrangianSize, lagrangianRoot, vertexIdentifierStyle) {}
#endif

  //! Destructor.
  virtual
  ~EulerianCommBoundary() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  //! Start sending the pressure to the relevant solid processors.
  /*!
    Call this function after the pressures at the nodes have been set.
  */
  virtual
  void
  sendPressure();

  //! Wait for the pressure sends to be copied into communication buffers.
  /*!
    This function must be called after sendPressure().
  */
  virtual
  void
  waitForPressure();

  // @}

protected:

  //! Initialize the pressure at the nodes.
  virtual
  void
  initializePressure();

};


} // namespace elc
}

#define __elc_EulerianCommBoundary_ipp__
#include "stlib/elc/EulerianCommBoundary.ipp"
#undef __elc_EulerianCommBoundary_ipp__

#endif

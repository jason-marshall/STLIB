// -*- C++ -*-

/*!
  \file EulerianCommShell.h
  \brief Eulerian communicator for the Eulerian-Lagrangian coupling.
*/

#if !defined(__elc_EulerianCommShell_h__)
#define __elc_EulerianCommShell_h__

#include "stlib/elc/EulerianComm.h"

namespace stlib
{
namespace elc
{


//! Eulerian communicator for the Eulerian-Lagrangian coupling for a shell.
/*!
  \param N is the space dimension.  2 and 3 are supported.
  \param T is the floating point number type.  By default it is double.

  This class is instantiated in each Eulerian (fluid) processor.  The
  relevant portions of the solid shell are received from the
  solid processors with either
  receiveMesh(const BBox& domain)
  or
  receiveMesh(const Number* domain)
  followed by a call to
  waitForMesh().

  One can access the node positions with either getPositions() or
  getPositionsData(); the node velocities can be access with getVelocities() or
  getVelocitiesData(); the face normals and centroids can be accessed with
  getFaceNormals() and getFaceCentroids().
  For each face that lies within this processor's computational domain, it
  must set the pressure difference across the face.  (The face is in the
  computational domain if the centroid is in the domain.)  The pressure
  differences may be manipulated through getPressures() or getPressuresData().

  After the pressures have been set, they are sent to the relevant Lagrangian
  processors with sendPressure() and waitForPressure().
*/
template < std::size_t N, typename T = double >
class EulerianCommShell :
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
  //! The assembled connectivities.
  using Base::_assembledConnectivities;
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
  EulerianCommShell();

  // Copy constructor not implemented.
  EulerianCommShell(const EulerianCommShell&);

  // Assignment operator not implemented.
  EulerianCommShell&
  operator=(const EulerianCommShell&);

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
  EulerianCommShell(const MPI::Comm& comm, const MPI::Intracomm& eulerian,
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
  EulerianCommShell(const MPI_Comm comm, const MPI_Comm eulerian,
                    const int lagrangianSize, const int lagrangianRoot,
                    VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, eulerian, lagrangianSize, lagrangianRoot, vertexIdentifierStyle) {}
#endif

  //! Destructor.
  virtual
  ~EulerianCommShell() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  //! Start sending the pressure to the relevant solid processors.
  /*!
    Call this function after the pressure differences across the faces have
    been set.
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

  //! Initialize the pressure at the faces.
  virtual
  void
  initializePressure();

};


} // namespace elc
}

#define __elc_EulerianCommShell_ipp__
#include "stlib/elc/EulerianCommShell.ipp"
#undef __elc_EulerianCommShell_ipp__

#endif

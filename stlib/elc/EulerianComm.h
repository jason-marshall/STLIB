// -*- C++ -*-

/*!
  \file EulerianComm.h
  \brief Eulerian communicator for the Eulerian-Lagrangian coupling.
*/

#if !defined(__elc_EulerianComm_h__)
#define __elc_EulerianComm_h__

#include "stlib/elc/ELComm.h"

#if defined(ELC_USE_CPP_INTERFACE) && !defined(PT2PT_BBOX_USE_CPP_INTERFACE)
#define PT2PT_BBOX_USE_CPP_INTERFACE
#endif
#include "stlib/concurrent/pt2pt_bbox.h"

#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/ads/functor/select.h"

#include "stlib/geom/mesh/iss/set.h"

#include <set>
#include <map>

namespace stlib
{
namespace elc
{

//! Base class Eulerian communicator for the Eulerian-Lagrangian coupling.
/*!
  \param N is the space dimension.  2 and 3 are supported.
  \param T is the floating point number type.

  Implements the common functionality for boundaries and shells.
*/
template<std::size_t N, typename T>
class EulerianComm :
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

  //! The number type.
  typedef typename Base::Number Number;
  //! A Cartesian point.
  typedef typename Base::Point Point;
  //! A bounding box.
  typedef typename Base::BBox BBox;
  //! An MPI request.
  typedef typename Base::MpiRequest MpiRequest;
  //! Status for an MPI request.
  typedef typename Base::MpiStatus MpiStatus;
  //! An indexed face type.
  typedef typename Base::IndexedFace IndexedFace;

  //
  // Using base members.
  //

protected:

  //! The joint Eulerian/Lagrangian communicator.
  using Base::_comm;
  //! The MPI number type.
  using Base::_mpiNumber;
  //! The vertex identifier style.
  using Base::_vertexIdentifierStyle;

  //! The communication tag for pressures.
  using Base::TagPressures;

private:

  //! The communication tag for node identifiers.
  using Base::TagIdentifiers;
  //! The communication tag for node positions.
  using Base::TagPositions;
  //! The communication tag for node velocities.
  using Base::TagVelocities;
  //! The communication tag for face data.
  using Base::TagFaceData;

  //
  // Member data.
  //

protected:

#ifdef ELC_USE_CPP_INTERFACE
  //! The Eulerian communicator.
  MPI::Intracomm _eulerianCommunicator;
#else
  //! The Eulerian communicator.
  MPI_Comm _eulerianCommunicator;
#endif

  //! The Lagrangian root.
  int _lagrangianRoot;

  //! The node identifiers from Lagrangian processors.
  std::vector<std::vector<int> > _identifiers;
  //! The node positions from Lagrangian processors.
  std::vector<std::vector<Point> > _positions;
  //! The node velocities from Lagrangian processors.
  std::vector<std::vector<Point> > _velocities;
  //! The node connectivities from Lagrangian processors.
  std::vector<std::vector<IndexedFace> > _connectivities;
  //! The node pressures to be sent to Lagrangian processors.
  std::vector<std::vector<Number> > _pressures;

  //! The mapping from node identifiers to node indices in the assembled boundary.
  std::map<int, int> _identifierToIndex;

  // CONTINUE: use something derived from an indexed simplex set instead.
  // That would facilitate interpolation.
  //! The assembled positions.
  std::vector<Point> _assembledPositions;
  //! The assembled velocities.
  std::vector<Point> _assembledVelocities;
  //! The assembled connectivities.
  std::vector<IndexedFace> _assembledConnectivities;
  //! The assembled pressures.
  std::vector<Number> _assembledPressures;

  //! Class for computing the point-to-point communication scheme.
  concurrent::PtToPt2Grp1Dom<N, T, int, std::array<std::size_t, 3> >
  _pointToPoint;

  //! Data from the Lagrangian processors with which we communicate.
  std::vector<std::array<std::size_t, 3> > _lagrangianData;

  //! The face normals.
  std::vector<Point> _faceNormals;
  //! The face centroids.
  std::vector<Point> _faceCentroids;

private:

  std::vector<MpiRequest> _identifierRequests;
  std::vector<MpiRequest> _positionRequests;
  std::vector<MpiRequest> _velocityRequests;
  std::vector<MpiRequest> _connectivityRequests;

  //
  // Not implemented.
  //

private:

  // Default constructor not implemented.
  EulerianComm();

  // Copy constructor not implemented.
  EulerianComm(const EulerianComm&);

  // Assignment operator not implemented.
  EulerianComm&
  operator=(const EulerianComm&);

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
    \param lagrangian_size is the number of Lagrangian processors.
    \param lagrangian_root is the rank of the Lagrangian root in \c comm.
    \param vertexIdentifierStyle is either LocalIndices or GlobalIdentifiers.
  */
  EulerianComm(const MPI::Comm& comm, const MPI::Intracomm& eulerian,
               const int lagrangianSize, const int lagrangianRoot,
               VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, vertexIdentifierStyle),
    _eulerianCommunicator(eulerian.Dup()),
    _lagrangianRoot(lagrangianRoot),
    _pointToPoint(comm, eulerian, lagrangianSize, lagrangianRoot) {}
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
  EulerianComm(const MPI_Comm comm, const MPI_Comm eulerian,
               const int lagrangianSize, const int lagrangianRoot,
               VertexIdentifierStyle vertexIdentifierStyle) :
    Base(comm, vertexIdentifierStyle),
    _eulerianCommunicator(),
    _lagrangianRoot(lagrangianRoot),
    _pointToPoint(comm, eulerian, lagrangianSize, lagrangianRoot)
  {
    MPI_Comm_dup(eulerian, &_eulerianCommunicator);
  }
#endif

  //! Destructor.  Free the duplicated communicator.
  virtual
  ~EulerianComm()
  {
#ifdef ELC_USE_CPP_INTERFACE
    _eulerianCommunicator.Free();
#else
    MPI_Comm_free(&_eulerianCommunicator);
#endif
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the number of nodes.
  std::size_t
  getNumberOfNodes() const
  {
    return _assembledPositions.size();
  }

  //! Return the number of faces.
  std::size_t
  getNumberOfFaces() const
  {
    return _assembledConnectivities.size();
  }

  //! Return a const reference to the array of node positions.
  const std::vector<Point>&
  getPositions() const
  {
    return _assembledPositions;
  }

  //! Return a const pointer to the node positions data.
  const Number*
  getPositionsData() const
  {
    return &_assembledPositions[0][0];
  }

  //! Return a const reference to the array of node velocities.
  const std::vector<Point>&
  getVelocities() const
  {
    return _assembledVelocities;
  }

  //! Return a const pointer to the node velocities data.
  const Number*
  getVelocitiesData() const
  {
    return &_assembledVelocities[0][0];
  }

  //! Return a const reference to the array of node connectivities.
  const std::vector<IndexedFace>&
  getConnectivities() const
  {
    return _assembledConnectivities;
  }

  //! Return a const pointer to the node connectivities data.
  const int*
  getConnectivitiesData() const
  {
    return &_assembledConnectivities[0][0];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Face Centroid/Normal Accessors.
  // @{

  //! Return a const reference to the array of face normals.
  const std::vector<Point>&
  getFaceNormals() const
  {
    return _faceNormals;
  }

  //! Return a const pointer to the face normals data.
  const Number*
  getFaceNormalsData() const
  {
    return &_faceNormals[0][0];
  }

  //! Return a const reference to the n_th face normal.
  const Point&
  getFaceNormal(const std::size_t n) const
  {
    return _faceNormals[n];
  }

  //! Return a const pointer to the n_th face normal data.
  const Number*
  getFaceNormalData(const std::size_t n) const
  {
    return &_faceNormals[n][0];
  }

  //! Return a const reference to the array of face centroids.
  const std::vector<Point>&
  getFaceCentroids() const
  {
    return _faceCentroids;
  }

  //! Return a const pointer to the face centroids data.
  const Number*
  getFaceCentroidsData() const
  {
    return &_faceCentroids[0][0];
  }

  //! Return a const reference to the n_th face centroid.
  const Point&
  getFaceCentroid(const std::size_t n) const
  {
    return _faceCentroids[n];
  }

  //! Return a const pointer to the n_th face centroid data.
  const Number*
  getFaceCentroidData(const std::size_t n) const
  {
    return &_faceCentroids[n][0];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Return a reference to the array of node pressures.
  std::vector<Number>&
  getPressures()
  {
    return _assembledPressures;
  }

  //! Return a pointer to the node pressures data.
  Number*
  getPressuresData()
  {
    return &_assembledPressures[0];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  //! Post receives for the relevant portions of the mesh from the solid processors.
  /*!
    \param domain is the region of interest for this fluid processor.
  */
  void
  receiveMesh(const BBox& domain);

  //! Post receives for the relevant portions of the mesh from the solid processors.
  /*!
    The lower and upper corners define the region of interest for this fluid processor.

    This function just calls the above
    \c receiveMesh(const BBox& domain).
  */
  void
  receiveMesh(const Number* lowerCorner, const Number* upperCorner)
  {
    receiveMesh(BBox(ext::copy_array<Point>(lowerCorner),
                     ext::copy_array<Point>(upperCorner)));
  }

  //! Wait for the receives to complete.  Build the assembled mesh.
  /*!
    This function must be called before accessing the mesh.
  */
  void
  waitForMesh();


  //! Start sending the pressure to the relevant solid processors.
  /*!
    Call this function after the pressures have been set.
  */
  virtual
  void
  sendPressure() = 0;

  //! Wait for the pressure sends to be copied into communication buffers.
  /*!
    This function must be called after sendPressure().
  */
  virtual
  void
  waitForPressure() = 0;


  //! Compute the face normals.
  /*!
    Call this after waitForMesh() if you will use the face normals.
  */
  void
  computeFaceNormals();

  //! Compute the face centroids.
  /*!
    Call this after waitForMesh() if you will use the face centroids.
  */
  void
  computeFaceCentroids();

  //! Initialize the pressure at the nodes or the faces.
  /*!
    This function should be called after waitForMesh() and before accessing
    the pressures.

    This function is implemented in the derived classes.
  */
  virtual
  void
  initializePressure() = 0;

  // @}

private:

  //! Compute the n_th face normal.
  void
  computeFaceNormal(const std::size_t n, Point* normal) const;

  //! Compute the n_th face centroid.
  void
  computeFaceCentroid(const std::size_t n, Point* centroid) const;

  //! Generate identifiers for the pieces of the boundary.
  void
  generateIdentifiers();

};


} // namespace elc
}

#define __elc_EulerianComm_ipp__
#include "stlib/elc/EulerianComm.ipp"
#undef __elc_EulerianComm_ipp__

#endif

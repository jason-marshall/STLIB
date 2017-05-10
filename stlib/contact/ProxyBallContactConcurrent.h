// -*- C++ -*-

/*!
  \file contact/ProxyBallContactConcurrent.h
  \brief Report contact using the proxy ball method.
*/

#if !defined(__contact_ProxyBallContactConcurrent_h__)
#define __contact_ProxyBallContactConcurrent_h__

#include "stlib/contact/ProxyBallContact.h"

#include <set>

#include <mpi.h>

namespace stlib
{
namespace contact
{

//! Report restoring forces for contact using the proxy ball method.
/*!
  \c N is the space dimension. \c _Identifier is the identifier type. This
  must be assignable to \c std::size_t.
*/
template < std::size_t N, typename _Identifier = std::size_t >
class ProxyBallContactConcurrent : public ProxyBallContact<N, _Identifier>
{
  // Types.

private:

  typedef ProxyBallContact<N, _Identifier> Base;

public:

  //! The floating-point number type.
  typedef typename Base::Number Number;
  //! The identifier type for vertices and elements.
  typedef typename Base::Identifier Identifier;
  //! A Cartesian point.
  typedef typename Base::Point Point;
  //! A tuple of element index and force vector.
  typedef typename Base::Force Force;

protected:

  //! A proxy ball.
  typedef typename Base::ProxyBall ProxyBall;
  //! An iterator to a proxy ball.
  typedef typename Base::ProxyBallIterator ProxyBallIterator;
  //! An const iterator to a proxy ball.
  typedef typename Base::ProxyBallConstIterator ProxyBallConstIterator;
  //! A bounding box.
  typedef geom::BBox<N> BBox;

  // Enumerations.
protected:

  //! Tags for the different communications.
  enum {TagSize, TagProxyBalls, TagForces};

  // Member data.
protected:

  //! All processors.
  MPI::Intracomm _comm;
  //! The number of local proxy balls of each component.
  std::vector<std::size_t> _numberOfLocalProxyBallsInComponent;
  //! For each processor, the maximum proxy ball radius in each component.
  /*! This is computed at the beginning of the simulation and upon restart. */
  std::vector<std::vector<Number> > _maxRadii;
  //! For each processor: whether we are receiving proxy balls from them.
  std::vector<bool> _areReceivingProxyBallsFrom;
  //! For each processor: whether we are sending proxy balls to them.
  std::vector<bool> _areSendingProxyBallsTo;

  using Base::_numberOfComponents;
  using Base::_proxyBalls;
  using Base::_numberOfLocalProxyBalls;
  using Base::_components;

  // Not implemented.
private:

  ProxyBallContactConcurrent();
  ProxyBallContactConcurrent(const ProxyBallContactConcurrent&);
  ProxyBallContactConcurrent&
  operator=(const ProxyBallContactConcurrent&);

public:

  //! Constructor.
  /*!
    \param comm The intracommunicator for the group of all processors.

    \param numberOfComponents The number of connected components in the initial
    mesh configuration. Note that this could be computed from the array of
    component indices. It is included as a parameter for consistency with
    the concurrent version of this class.

    \param numberOfNodes The number of nodes.
    \param nodeCoordinates The coordinates of the nodes. Each coordinate is
    an N-tuple. The node coordinates are needed to compute the radii for the
    proxy balls.
    \param nodeIdentifiers The node identifiers.

    \param numberOfElements The number of elements.
    \param identifierSimplices The identifier simplices. Each element is
    represented by an (N+1)-tuple of node identifiers.
    \param components The index of the connected component to which
    each element belongs. A value greater than or equal to the number of
    components indicates an inactive element.

    \param maximumRelativePenetration is the maximum allowed relative
    penetration of the balls.
    \param springFraction is the fraction of the spring force to use. By
    default it is 1.
    \param dampingFraction is the fraction of the damping force to use. By
    default it is 0.
   */
  ProxyBallContactConcurrent(const MPI::Intracomm& comm,
                             const std::size_t numberOfComponents,
                             const std::size_t numberOfNodes,
                             const Number* nodeCoordinates,
                             const Identifier* nodeIdentifiers,
                             const std::size_t numberOfElements,
                             const Identifier* identifierSimplices,
                             const std::size_t* components,
                             const Number maximumRelativePenetration,
                             const Number springFraction = 1,
                             const Number dampingFraction = 0);

  //! Report restoring forces for contact using the proxy ball method.
  /*!
    \param nodeCoordinates The coordinates of the nodes. Each coordinate is
    an N-tuple.
    \param velocityCoordinates The velocities of the nodes. Each velocity is
    an N-tuple.
    \param masses The element masses.
    \param components The index of the connected component to which
    each element belongs. A value greater than or equal to the number of
    components indicates an inactive element.

    \return Stable time step for contact. */
  template<typename _OutputIterator>
  Number
  operator()(const Number* nodeCoordinates,
             const Number* velocityCoordinates,
             const Number* masses,
             const std::size_t* components,
             _OutputIterator elementForces)
  {
    return operator()(nodeCoordinates, velocityCoordinates, masses, components,
                      elementForces,
                      ads::constructTrivialOutputIterator(),
                      ads::constructTrivialOutputIterator());
  }

  //! Report restoring forces for contact using the proxy ball method.
  /*!
    \param nodeCoordinates The coordinates of the nodes. Each coordinate is
    an N-tuple.
    \param velocityCoordinates The velocities of the nodes. Each velocity is
    an N-tuple.
    \param masses The element masses.
    \param components The index of the connected component to which
    each element belongs. A value greater than or equal to the number of
    components indicates an inactive element.

    \return Stable time step for contact.

    The forces are reported as a std::tuple of
    (simplexIdentifier, force). The force is represented with a
    std::array<double,N>.

    The number of interactions for each element in contact is recorded in
    \c interactionCounts. A relative penetration is reported for each contact.
  */
  template < typename _ForceOutputIterator, typename _CountOutputIterator,
             typename _RelPenOutputIterator >
  Number
  operator()(const Number* nodeCoordinates,
             const Number* velocityCoordinates,
             const Number* masses,
             const std::size_t* components,
             _ForceOutputIterator elementForces,
             _CountOutputIterator interactionCounts,
             _RelPenOutputIterator relativePenetrations);

protected:

  //! Set the processor ranks and element identifiers for each proxy ball.
  void
  computeGlobalIdentifiers();

  //! Compute the maximum radius in each component.
  void
  computeMaxRadii();

  //! Exchange the proxy balls using a point-to-point communication pattern.
  void
  exchangeProxyBalls(const std::size_t* components);

  //! Exchange the forces using a point-to-point communication pattern.
  /*! Record the local forces in elementForces. */
  template<typename _ForceOutputIterator>
  void
  exchangeForces(const std::vector<std::tuple<std::size_t, Point> >&
                 allForces, _ForceOutputIterator elementForces);

  //! Purge the ghost proxy balls.
  void
  purgeGhosts();

  //! Expand the bounding box by twice the maximum radius in the specified component.
  void
  expandBoundingBox(BBox* box, const std::size_t rank,
                    const std::size_t component) const;
};

} // namespace contact
}

#define __contact_ProxyBallContactConcurrent_ipp__
#include "stlib/contact/ProxyBallContactConcurrent.ipp"
#undef __contact_ProxyBallContactConcurrent_ipp__

#endif

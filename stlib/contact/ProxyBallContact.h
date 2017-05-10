// -*- C++ -*-

/*!
  \file contact/ProxyBallContact.h
  \brief Report contact using the proxy ball method.
*/

#if !defined(__contact_ProxyBallContact_h__)
#define __contact_ProxyBallContact_h__

#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/mesh/iss/topology.h"
#include "stlib/geom/mesh/simplex/geometry.h"
#include "stlib/geom/kernel/content.h"
#include "stlib/geom/orq/CellArrayStatic.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/ext/pair.h"
#include "stlib/numerical/constants.h"

#include <functional>
#include <tuple>
#include <unordered_map>

namespace std
{
template<>
struct hash<array<std::size_t, 2> > {
  std::size_t
  operator()(const array<std::size_t, 2>& x) const
  {
    hash<std::size_t> h;
    return h(x[0] + x[1]);
  }
};
}

namespace stlib
{
namespace contact
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! A ball that serves as a proxy in detecting contact.
template<std::size_t N>
struct ProxyBall {
  // Types.

  //! The floating-point number type.
  typedef double Number;
  //! A Cartesian point.
  typedef std::array<Number, N> Point;

  // Mutable data.

  //! The center of the ball.
  Point center;
  //! The velocity of the associated element.
  Point velocity;
  //! The mass of the associated element.
  Number mass;

  // Constant data.

  //! The radius of the ball.
  Number radius;
  //! The index of the mesh component to which the associated element belongs.
  std::size_t component;
  //! The processor to which the associated element belongs.
  std::size_t processor;
  //! The index of the associated element.
  std::size_t elementIndex;
  //! The identifier of the associated element.
  std::size_t elementIdentifier;
};

//! Write information about the proxy ball.
/*! \relates ProxyBall */
template<std::size_t N>
std::ostream&
operator<<(std::ostream& out, const ProxyBall<N>& x);

//! Equality.
/*! \relates ProxyBall */
template<std::size_t N>
bool
operator==(const ProxyBall<N>& x, const ProxyBall<N>& y);

//! Functor for accessing the center of a proxy ball from an iterator to an iterator to a proxy ball.
template<typename _Iterator, typename _Point>
struct GetProxyBallCenter :
    public std::unary_function<_Iterator, _Point> {
  typedef std::unary_function<_Iterator, _Point> Base;
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;
  //! Return the center.
  const result_type&
  operator()(argument_type x) const
  {
    return (*x)->center;
  }
};

//! Report restoring forces for contact using the proxy ball method.
/*!
  \c N is the space dimension. \c _Identifier is the identifier type. This
  must be assignable to \c std::size_t.

  Let <em>m</em> denote the mass of a ball and <em>v</em> denote the
  component of the velocity
  along the restoring force direction. We use this to compute a kinetic
  energy: \f$0.5 m_1 v_1^2 + 0.5 m_2 v_2^2\f$.
  Let <em>k</em> be the spring constant and <em>d</em> be the maximum
  allowed penetration,
  which is the maximum allowed relative penetration times the sum of the
  ball radii. At the maximum penetration the potential energy in the spring
  is \f$0.5 k d^2\f$. We equate the kinetic and potential energy to
  determine the spring constant.
  \f[
  k = \frac{m_1 v_1^2 + m_2 v_2^2}{d^2}
  \f]
  The restoring force from the spring is <em>k</em> times the penetration.

  Damping forces can be modeled with the problem,
  \f[
  m x'' = -b x', \quad x(0) = 0, \quad x'(0) = - v,
  \f]
  which has the solution
  \f[
  x = \frac{m v}{b} \left(\mathrm{e}^{-b t/m} - 1\right).
  \f]
  This means that the damping force stops the mass at a displacement of
  <em>m v / b</em>. Equating this displacement to maximum allowed displacement
  yields the damping constant <em>b = m v / d</em>.

  If the damping fraction is nonzero, there will be damping forces
  applied. The damping coefficient is \f$b = (m_1 + m_2)|v_1 - v_2| / d\f$.
  The damping force is <em>b</em> times the relative velocity between
  the balls.
*/
template < std::size_t N, typename _Identifier = std::size_t >
class ProxyBallContact
{
  // Types.
public:

  //! The floating-point number type.
  typedef double Number;
  //! The identifier type for vertices and elements.
  typedef _Identifier Identifier;
  //! A Cartesian point.
  typedef std::array<Number, N> Point;
  //! A tuple of element index and force vector.
  typedef std::tuple<std::size_t, Point> Force;

protected:

  //! The hash table type used to store spring constants.
  typedef std::unordered_map < std::array<std::size_t, 2>,
          std::pair<Number, Number> > HashTable;
  //! A simplex of vertices.
  typedef std::array < Point, N + 1 > Simplex;
  //! A proxy ball.
  typedef contact::ProxyBall<N> ProxyBall;
  //! An iterator to a proxy ball.
  typedef typename std::vector<ProxyBall>::iterator ProxyBallIterator;
  //! An const iterator to a proxy ball.
  typedef typename std::vector<ProxyBall>::const_iterator
  ProxyBallConstIterator;

  // Member data.
protected:

  //! The number of connected components in the initial mesh.
  std::size_t _numberOfComponents;
  //! The number of nodes.
  std::size_t _numberOfNodes;
  //! The number of elements.
  std::size_t _numberOfElements;
  //! The index simplices. Each element is represented by an (N+1)-tuple of node indices. (Note: indices not identifiers.)
  std::vector < std::array < std::size_t, N + 1 > > _indexSimplices;
  //! The proxy balls for each element.
  std::vector<ProxyBall> _proxyBalls;
  //! The number of local proxy balls.
  std::size_t _numberOfLocalProxyBalls;
  //! Iterators to the active proxy balls, grouped by component.
  std::vector<std::vector<ProxyBallIterator> > _components;

  //! The target maximum relative penetration.
  Number _maximumRelativePenetration;
  //! The fraction of the spring force to use.
  Number _springFraction;
  //! The fraction of the damping force to use.
  Number _dampingFraction;
  //! Hash table of the spring and damping constants for the active contact events.
  HashTable _springAndDampingConstants;

  // Not implemented.
private:

  ProxyBallContact();
  ProxyBallContact(const ProxyBallContact&);
  ProxyBallContact&
  operator=(const ProxyBallContact&);

public:

  //! Constructor.
  /*!
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

    \param maximumRelativePenetration is the maximum allowed relative
    penetration of the balls.
    \param springFraction is the fraction of the spring force to use. By
    default it is 1.
    \param dampingFraction is the fraction of the damping force to use. By
    default it is 0.
   */
  ProxyBallContact(const std::size_t numberOfComponents,
                   const std::size_t numberOfNodes,
                   const Number* nodeCoordinates,
                   const Identifier* nodeIdentifiers,
                   const std::size_t numberOfElements,
                   const Identifier* identifierSimplices,
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

  //! Write the state.
  template<std::size_t N_, typename _Identifier_>
  friend
  std::ostream&
  operator<<(std::ostream& out, const ProxyBallContact<N_, _Identifier_>& x);

  //! Read the state.
  template<std::size_t N_, typename _Identifier_>
  friend
  std::istream&
  operator>>(std::istream& in, ProxyBallContact<N_, _Identifier_>& x);

  //! Equality.
  template<std::size_t N_, typename _Identifier_>
  friend
  bool
  operator==(const ProxyBallContact<N_, _Identifier_>& x,
             const ProxyBallContact<N_, _Identifier_>& y);

protected:

  //! Initialize the data for the proxy balls in preparation for a step.
  void
  initializeProxyBalls(const Number* nodeCoordinates,
                       const Number* velocityCoordinates, const Number* masses,
                       const std::size_t* components);

  //! Compute the forces to apply at the element centroids.
  template < typename _ForceOutputIterator, typename _CountOutputIterator,
             typename _RelPenOutputIterator >
  void
  computeForces(const std::vector < std::tuple < std::size_t, std::size_t,
                Point > > penetrations,
                _ForceOutputIterator elementForces,
                _CountOutputIterator interactionCounts,
                _RelPenOutputIterator relativePenetrations);

  //! Report contact using the pinball method.
  /*!
    \param contacts The output iterator for the contacts.

    The contacts are reported as a std::tuple of
    (index0, index1, penetration). The indices are the
    simplex indices. The penetration is a vector whose length is the
    penetration distance and whose direction is from the centroid of the
    latter simplex toward the centroid of the former simplex.
    The penetration vector is represented with a std::array<Number,N>.

    Each contact is reported only once. A contact between two simplices is
    reported only if the simplices are in different components.
  */
  template<typename _OutputIterator>
  void
  computeContact(_OutputIterator contacts);

  //! Compute the stable time step.
  /*! Check for the worst case scenario of each element colliding head-on with
    a mirror image of itself.*/
  Number
  computeStableTimeStep(const std::size_t* components) const;

  //! Compute the radii of the proxy balls.
  /*!
    The radius of the ball is determined by the initial configuration of the
    associated element. The ball encloses the midpoint nodes.
  */
  void
  computeRadii(const Number* nodeCoordinates);

  //! Compute the simplex for the specified element.
  void
  computeSimplex(Simplex* s, const Number* nodeCoordinates,
                 const std::size_t i) const;

  //! Compute the velocity for the specified element.
  void
  computeVelocity(Point* velocity, const Number* velocityCoordinates,
                  const std::size_t i);

  //! Count the number of components with active elements.
  std::size_t
  countComponents() const;
};

//! Write the radii of the proxy balls. Write the spring and damping constants.
/*!
  \relates ProxyBallContact

  Format:
  \verbatim
  numberOfElements
  radius0
  radius1
  ...
  numberOfCachedConstants
  sourceIdentifier0 targetIdentifier0 springConstant0 dampingConstant0
  sourceIdentifier1 targetIdentifier1 springConstant1 dampingConstant1
  ... \endverbatim
*/
template<std::size_t N, typename _Identifier>
std::ostream&
operator<<(std::ostream& out, const ProxyBallContact<N, _Identifier>& x);

//! Read the proxy ball radii and the spring constants.
/*! \relates ProxyBallContact */
template<std::size_t N, typename _Identifier>
std::istream&
operator>>(std::istream& in, ProxyBallContact<N, _Identifier>& x);


//! Equality.
/*! \relates ProxyBallContact */
template<std::size_t N, typename _Identifier>
bool
operator==(const ProxyBallContact<N, _Identifier>& x,
           const ProxyBallContact<N, _Identifier>& y);

} // namespace contact
}

#define __contact_ProxyBallContact_ipp__
#include "stlib/contact/ProxyBallContact.ipp"
#undef __contact_ProxyBallContact_ipp__

#endif

// -*- C++ -*-

/*!
  \file contact/pinballContact.h
  \brief Report contact using the pinball method.
*/

#if !defined(__contact_pinballContact_h__)
#define __contact_pinballContact_h__

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

//-----------------------------------------------------------------------------
/*! \defgroup iss_pinballContact Report contact using the pinball method.
*/
//@{

//! Report restoring forces for contact using the pinball method.
/*!
  The forces are reported as a std::tuple of
  (simplexIndex, force). The force is represented with a std::array<T,N>.
*/
template < std::size_t N, typename T = double >
class PinballRestoringForces
{
  // Types.
public:

  //! The hash table type used to store spring constants.
  typedef std::unordered_map < std::array<std::size_t, 2>,
          std::pair<T, T> > HashTable;

  // Member data.
private:

  T _maximumRelativePenetration;
  T _springFraction;
  T _dampingFraction;
  HashTable _springAndDampingConstants;

  // Not implemented.
private:

  PinballRestoringForces();
  PinballRestoringForces(const PinballRestoringForces&);
  PinballRestoringForces&
  operator=(const PinballRestoringForces&);

public:

  //! Constructor.
  /*!
    \param maximumRelativePenetration is the maximum allowed relative
    penetration of the balls.
    \param springFraction is the fraction of the spring force to use. By
    default it is 1.
    \param dampingFraction is the fraction of the damping force to use. By
    default it is 0.
   */
  PinballRestoringForces(const T maximumRelativePenetration,
                         const T springFraction = 1,
                         const T dampingFraction = 0) :
    _maximumRelativePenetration(maximumRelativePenetration),
    _springFraction(springFraction),
    _dampingFraction(dampingFraction),
    _springAndDampingConstants()
  {
    assert(maximumRelativePenetration > 0);
  }

  //! Report restoring forces for contact using the pinball method.
  /*! \return Stable time step for contact. */
  template<typename _OutputIterator>
  T
  operator()(const std::vector<std::array<T, N> >& vertices,
             const std::vector<std::size_t>& vertexIdentifiers,
             const std::vector<std::array<T, N> >& velocities,
             const std::vector < std::array < std::size_t, N + 1 > > &
             identifierSimplices,
             const std::vector<T>& masses,
             _OutputIterator elementForces)
  {
    return operator()(vertices, vertexIdentifiers, velocities,
                      identifierSimplices, masses, elementForces,
                      ads::constructTrivialOutputIterator(),
                      ads::constructTrivialOutputIterator());
  }

  //! Report restoring forces for contact using the pinball method.
  /*! \return Stable time step for contact.

    The number of interactions for each element in contact is recorded in
    \c interactionCounts. A relative penetration is reported for each contact.

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
  template < typename _ForceOutputIterator, typename _CountOutputIterator,
             typename _RelPenOutputIterator >
  T
  operator()(const std::vector<std::array<T, N> >& vertices,
             const std::vector<std::size_t>& vertexIdentifiers,
             const std::vector<std::array<T, N> >& velocities,
             const std::vector < std::array < std::size_t, N + 1 > > &
             identifierSimplices,
             const std::vector<T>& masses,
             _ForceOutputIterator elementForces,
             _CountOutputIterator interactionCounts,
             _RelPenOutputIterator relativePenetrations);

  //! Return a const reference to the spring and damping constants.
  const HashTable&
  getSpringAndDampingConstants() const
  {
    return _springAndDampingConstants;
  }

  //! Return a reference to the spring and damping constants.
  HashTable&
  getSpringAndDampingConstants()
  {
    return _springAndDampingConstants;
  }
};

//! Write the spring constants.
/*!
  \relates PinballRestoringForces

  Format:
  \verbatim
  size
  sourceIdentifier0 targetIdentifier0 springConstant0 dampingConstant0
  sourceIdentifier1 targetIdentifier1 springConstant1 dampingConstant1
  ... \endverbatim
*/
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const PinballRestoringForces<N, T>& x)
{
  using stlib::ext::operator<<;

  out << x.getSpringAndDampingConstants().size() << '\n';
  for (auto i = x.getSpringAndDampingConstants().begin();
       i != x.getSpringAndDampingConstants().end(); ++i) {
    out << *i << '\n';
  }
  return out;
}

//! Read the spring constants.
/*!
  \relates PinballRestoringForces
*/
template<std::size_t N, typename T>
inline
std::istream&
operator>>(std::istream& in, PinballRestoringForces<N, T>& x)
{
  USING_STLIB_EXT_PAIR_IO_OPERATORS;

  typedef typename PinballRestoringForces<N, T>::HashTable HashTable;
  typedef typename HashTable::value_type value_type;
  typedef typename HashTable::key_type key_type;
  typedef typename HashTable::mapped_type mapped_type;

  // Clear the any old spring constants.
  x.getSpringAndDampingConstants().clear();
  // Get the number of spring constants.
  std::size_t size;
  in >> size;
  // Read each of them.
  key_type key;
  mapped_type mapped;
  for (size_t i = 0; i != size; ++i) {
    in >> key >> mapped;
    x.getSpringAndDampingConstants().insert(value_type(key, mapped));
  }
  return in;
}

//! Report contact using the pinball method.
template<std::size_t N, typename T, typename OutputIterator>
T
pinballContact(const std::vector<std::array<T, N> >& vertices,
               const std::vector<std::size_t>& vertexIdentifiers,
               const std::vector<std::array<T, N> >& velocities,
               const std::vector < std::array < std::size_t, N + 1 > > & simplices,
               T maximumRelativePenetration,
               OutputIterator contacts);

//! Report contact using the pinball method.
/*!
  \param mesh The mesh on which to perform pinball contact.
  \param velocities The node velocities.
  \param maximumRelativePenetration The maximum allowed penetration.
  \param contacts The output iterator for the contacts.

  The contacts are reported as a std::tuple of
  (index0, index1, penetration, sumOfRadii). The indices are the
  simplex indices. The penetration is a vector whose length is the
  penetration distance and whose direction is from the centroid of the
  latter simplex toward the centroid of the former simplex.
  The penetration vector is represented with a std::array<T,N>. The final
  field is the sum of the radii of the two balls.

  Each contact is reported only once. A contact between two simplices is
  reported only if the simplices do not share a vertex.

  \relates IndSimpSetIncAdj
*/
template<std::size_t N, typename T, typename OutputIterator>
T
pinballContact(const geom::IndSimpSetIncAdj<N, N, T>& mesh,
               const std::vector<std::array<T, N> >& velocities,
               T maximumRelativePenetration,
               OutputIterator contacts);

//@}

} // namespace contact
}

#define __contact_pinballContact_ipp__
#include "stlib/contact/pinballContact.ipp"
#undef __contact_pinballContact_ipp__

#endif

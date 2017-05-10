// -*- C++ -*-

/*!
  \file SimplexModDet.h
  \brief Implements operations for modifying the determinant of the Jacobian.
*/

#if !defined(__geom_SimplexModDet_h__)
#define __geom_SimplexModDet_h__

#include <limits>

#include <cmath>
#include <cassert>

namespace stlib
{
namespace geom {

//! Implements operations for modifying the determinant of the Jacobian.
/*!
  \param T is the number type.  By default it is double.

  This class cannot be constructed.  Its member data and public member
  functions are static.

  <b>The Modified Determinant</b>

  See the documentation of the geom::SimplexJac class for information on
  the Jacobian matrix of a simplex.
  When the content of a simplex vanishes, its Jacobian matrix becomes
  singular.  That is, its determinant vanishes.  This presents a problem
  for some algebraic quality metrics.  The condition number metric and the
  mean ratio metric (implemented in geom::SimplexCondNum and
  geom::SimplexMeanRatio) become singular as the content vanishes.
  In order to define these metrics for simplices with vanishing and
  negative content, we use a modified value \f$ h \f$ of the Jacobian
  determinant \f$ \sigma \f$ which has the following properties:
  - \f$ h \f$ is close to \f$ \sigma \f$ when
  \f$ \sigma \f$ is positive and \f$ \sigma \gg 1 \f$.
  - \f$ h \f$ is small and positive when the determinant is negative.
  - \f$ h \f$ is differentiable.

  Let \f$ \epsilon \f$ be a number that is a little bigger than the
  machine precision.  (We use 100 times the machine precision.)
  Define
  \f[
  \delta = \sqrt{ \epsilon (\epsilon - \sigma) }.
  \f]
  The modified value of the determinant is
  \f[
  h = \frac{ \sigma + \sqrt{ \sigma^2 + 4 \delta^2 } }{ 2 }.
  \f]

  <b>Usage</b>

  Consider a complex of simplices.  (Perhaps the tetrahedra that are
  adjacent to a vertex.)  Let \c minDeterminant be the minimum determinant
  of the simplices in the complex and \c determinant be the determinant
  of a given simplex.
  \c h(determinant,minDeterminant) returns the modified determinant.
  If the minimum determinant is no less than \f$ \epsilon \f$ then
  \c h() returns the un-modified determinant.
*/
template < typename T = double >
class SimplexModDet {
public:

   //
   // Public types.
   //

   //! The number type.
   typedef T Number;

private:

   //
   // Not implemented.
   //

   // Default constructor.
   SimplexModDet();

   // Copy constructor.
   SimplexModDet(const SimplexModDet&);

   // Assignment operator.
   SimplexModDet&
   operator=(const SimplexModDet&);

   // Destructor.
   ~SimplexModDet();

public:

   //--------------------------------------------------------------------------
   //! \name Mathematical functions
   //! @{

   //! Return epsilon.
   static
   Number
   getEpsilon() {
      // CONTINUE
      // This may be too conservative.
      // However, if I use 10 times the machine precision
      // the modified eta function overflows for some inverted simplices.
      return std::sqrt(std::numeric_limits<Number>::epsilon());
      //return 100.0 * std::numeric_limits<Number>::epsilon();
   }

   //! Return delta.
   static
   Number
   getDelta(const Number minDeterminant) {
      if (minDeterminant < getEpsilon()) {
         const Number argument = getEpsilon() * (getEpsilon() - minDeterminant);
#ifdef STLIB_DEBUG
         assert(argument >= 0);
#endif
         return std::sqrt(argument);
      }
      return 0.0;
   }

   //! Return a number that is close to the determinant when it is positive and small and positive when the determinant is negative.
   /*!
     \return
     Let \f$ \epsilon \f$ be the value of epsilon() and
     \f$ \sigma \f$ be the Jacobian determinant.
     Define
     \f[ \delta = \sqrt{ \epsilon (\epsilon - \sigma) }. \f]
     Return
     \f[ \frac{ \sigma + \sqrt{ \sigma^2 + 4 \delta^2 } }{ 2 }. \f]
   */
   static
   Number
   getH(Number determinant, Number minDeterminant);

   //! @}
};

} // namespace geom
}

#define __geom_SimplexModDet_ipp__
#include "stlib/geom/mesh/simplex/SimplexModDet.ipp"
#undef __geom_SimplexModDet_ipp__

#endif

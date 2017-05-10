// -*- C++ -*-

/*!
  \file Distance2.h
  \brief Finite difference operations for computing distance in 2-D.
*/

#if !defined(__hj_Distance2_h__)
#error This file is an implementation detail of the class Distance.
#endif

namespace stlib
{
namespace hj {

//! Finite differences for computing distance.
/*!
  \param T is the number type.

  Finite differences for computing distance by solving the eikonal
  equation \f$ | \nabla u | = 1 \f$.

  This class does not know anything about the solution or status
  grids.  This class defines protected member functions that perform
  finite difference operations.  It provides functionality for both
  adjacent and adjacent-diagonal difference schemes.  Classes that
  derive from \c Distance call these low-level functions.
*/
template<typename T>
class Distance<2, T> {






   //
   // Public types
   //

public:

   //! The number type.
   typedef T Number;

   //
   // Member data
   //

protected:

   //! The grid spacing.
   Number _dx;

   //! The grid spacing squared.
   Number _dx2;

   //! dx * sqrt(2)
   Number _dx_t_sqrt2;

   //! dx / sqrt(2)
   Number _dx_o_sqrt2;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   Distance();
   //! Copy constructor not implemented.
   Distance(const Distance&);
   //! Assignment operator not implemented.
   Distance&
   operator=(const Distance&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Constructor.
   /*!
     \param dx is the grid spacing.
   */
   Distance(Number dx);

   //@}

protected:

   //--------------------------------------------------------------------------
   //! \name Finite difference schemes.
   //@{

   //! First order in an adjacent direction.
   /*!
     \param a is the adjacent solution.
   */
   Number
   diff_a1(const Number a) const {
      return a + _dx;
   }

   //! First order in a diagonal direction.
   /*!
     \param d is the diagonal solution.
   */
   Number
   diff_d1(const Number d) const {
      return d + _dx_t_sqrt2;
   }

   //! Second order in an adjacent direction.
   /*!
     \param a1 is the adjacent solution.
     \param a2 is the adjacent solution which is two grid points away.
   */
   Number
   diff_a2(Number a1, Number a2) const;

   //! Second order in a diagonal direction.
   /*!
     \param d1 is the diagonal solution.
     \param d2 is the diagonal solution which is two grid points away.
   */
   Number
   diff_d2(Number d1, Number d2) const;

   //! First order adjacent. First order adjacent.
   Number
   diff_a1_a1(Number a, Number b) const;

   //! First order adjacent. First order diagonal.
   Number
   diff_a1_d1(Number a, Number d) const;


   //! Second order adjacent.  First order adjacent.
   Number
   diff_a2_a1(Number a1, Number a2, Number b) const;

   //! First order adjacent. Second order diagonal.
   Number
   diff_a1_d2(Number a, Number d1, Number d2) const;

   //! Second order adjacent. First order diagonal.
   Number
   diff_a2_d1(Number a1, Number a2, Number d) const;


   //! Second order adjacent. Second order Adjacent.
   Number
   diff_a2_a2(Number a1, Number a2,
              Number b1, Number b2) const;

   //! Second order adjacent. Second order diagonal.
   Number
   diff_a2_d2(Number a1, Number a2,
              Number d1, Number d2) const;

   //@}
};

} // namespace hj
}

#define __hj_Distance2_ipp__
#include "stlib/hj/Distance2.ipp"
#undef __hj_Distance2_ipp__

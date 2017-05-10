// -*- C++ -*-

/*!
  \file Eikonal2.h
  \brief Finite difference operations for the eikonal equation in 2-D.
*/

#if !defined(__hj_Eikonal2_h__)
#error This file is an implementation detail of the class Eikonal.
#endif

namespace stlib
{
namespace hj {

//! Finite differences for the eikonal equation \f$ | \nabla u | f = 1 \f$.
/*!
  \param T is the number type.

  This class does not know anything about the solution or status
  grids.  This class defines protected member functions that perform
  finite difference operations.  It provides functionality for both
  adjacent and adjacent-diagonal difference schemes.  Classes that
  derive from \c Eikonal call these low-level functions.
*/
template<typename T>
class Eikonal<2, T> {
public:

   //
   // Public types
   //

   //! The number type.
   typedef T Number;

protected:

   //
   // Member data
   //

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
   Eikonal();
   //! Copy constructor not implemented.
   Eikonal(const Eikonal&);
   //! Assignment operator not implemented.
   Eikonal&
   operator=(const Eikonal&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Constructor.
   /*!
     \param dx is the grid spacing.
   */
   Eikonal(Number dx);

   //@}

protected:

   //--------------------------------------------------------------------------
   //! \name Finite difference schemes.
   //@{

   //! First order in an adjacent direction.
   /*!
     \param a is the adjacent solution.
     \param f is the value of the speed function.
   */
   Number
   diff_a1(const Number a, const Number f) const {
      return a + _dx * f;
   }

   //! First order in a diagonal direction.
   /*!
     \param d is the diagonal solution.
     \param f is the value of the speed function.
   */
   Number
   diff_d1(const Number d, const Number f) const {
      return d + _dx_t_sqrt2 * f;
   }

   //! Second order in an adjacent direction.
   /*!
     \param a1 is the adjacent solution.
     \param a2 is the adjacent solution which is two grid points away.
     \param f is the value of the speed function.
   */
   Number
   diff_a2(Number a1, Number a2, Number f) const;

   //! Second order in a diagonal direction.
   /*!
     \param d1 is the diagonal solution.
     \param d2 is the diagonal solution which is two grid points away.
     \param f is the value of the speed function.
   */
   Number
   diff_d2(Number d1, Number d2, Number f) const;

   //! First order adjacent. First order adjacent.
   Number
   diff_a1_a1(Number a, Number b, Number f) const;

   //! First order adjacent. First order diagonal.
   Number
   diff_a1_d1(Number a, Number d, Number f) const;


   //! Second order adjacent.  First order adjacent.
   Number
   diff_a2_a1(Number a1, Number a2, Number b, Number f) const;

   //! First order adjacent. Second order diagonal.
   Number
   diff_a1_d2(Number a, Number d1, Number d2, Number f) const;

   //! Second order adjacent. First order diagonal.
   Number
   diff_a2_d1(Number a1, Number a2, Number d, Number f) const;


   //! Second order adjacent. Second order Adjacent.
   Number
   diff_a2_a2(Number a1, Number a2, Number b1, Number b2, Number f) const;

   //! Second order adjacent. Second order diagonal.
   Number
   diff_a2_d2(Number a1, Number a2, Number d1, Number d2, Number f) const;

   //@}
};

} // namespace hj
}

#define __hj_Eikonal2_ipp__
#include "stlib/hj/Eikonal2.ipp"
#undef __hj_Eikonal2_ipp__

// -*- C++ -*-

/*!
  \file Distance3.h
  \brief Finite difference operations for computing distance in 3-D.
*/

#if !defined(__hj_Distance3_h__)
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
class Distance<3, T> : public Distance<2, T> {
   //
   // Private types.
   //

   typedef Distance<2, T> Base;

   //
   // Public types.
   //

public:

   //! The number type.
   typedef T Number;

   //
   // Member data
   //

protected:

   //! The grid spacing.
   using Base::_dx;

   //! The grid spacing squared.
   using Base::_dx2;

   //! dx * sqrt(2)
   using Base::_dx_t_sqrt2;

   //! dx / sqrt(2)
   using Base::_dx_o_sqrt2;

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
   Distance(const Number dx) :
      Base(dx) {}

   //@}

protected:

   //--------------------------------------------------------------------------
   //! \name Finite difference schemes.
   //@{

   //! First order in an adjacent direction.
   using Base::diff_a1;

   //! First order in a diagonal direction.
   using Base::diff_d1;

   //! Second order in an adjacent direction.
   using Base::diff_a2;

   //! Second order in a diagonal direction.
   using Base::diff_d2;

   //! First order adjacent. First order adjacent.
   using Base::diff_a1_a1;

   //! First order adjacent. First order diagonal.
   using Base::diff_a1_d1;

   //! First order diagonal. First order diagonal.
   Number
   diff_d1_d1(Number a, Number b) const;


   //! Second order adjacent.  First order adjacent.
   using Base::diff_a2_a1;

   //! First order adjacent. Second order diagonal.
   using Base::diff_a1_d2;

   //! Second order adjacent. First order diagonal.
   using Base::diff_a2_d1;


   //! Second order adjacent. Second order Adjacent.
   using Base::diff_a2_a2;

   //! Second order adjacent. Second order diagonal.
   using Base::diff_a2_d2;


   //! First order adjacent. First order adjacent.  First order adjacent.
   Number
   diff_a1_a1_a1(Number a, Number b, Number c) const;

   //! First order adjacent. First order diagonal.  First order diagonal.
   Number
   diff_a1_d1_d1(Number a, Number b, Number c) const;

   //! First order diagonal. First order diagonal.  First order diagonal.
   Number
   diff_d1_d1_d1(Number a, Number b, Number c) const;

};

} // namespace hj
}

#define __hj_Distance3_ipp__
#include "stlib/hj/Distance3.ipp"
#undef __hj_Distance3_ipp__

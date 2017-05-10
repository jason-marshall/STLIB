// -*- C++ -*-

/*!
  \file Eikonal3.h
  \brief Finite difference operations for the eikonal equation in 3-D.
*/

#if !defined(__hj_Eikonal3_h__)
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
class Eikonal<3, T> : public Eikonal<2, T> {
   //
   // Private types.
   //

   typedef Eikonal<2, T> Base;

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
   Eikonal(const Number dx) :
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
   diff_d1_d1(Number a, Number b, Number f) const;


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
   diff_a1_a1_a1(Number a, Number b, Number c, Number f) const;

   //! First order adjacent. First order diagonal.  First order diagonal.
   Number
   diff_a1_d1_d1(Number a, Number b, Number c, Number f) const;

   //! First order diagonal. First order diagonal.  First order diagonal.
   Number
   diff_d1_d1_d1(Number a, Number b, Number c, Number f) const;

};

} // namespace hj
}

#define __hj_Eikonal3_ipp__
#include "stlib/hj/Eikonal3.ipp"
#undef __hj_Eikonal3_ipp__

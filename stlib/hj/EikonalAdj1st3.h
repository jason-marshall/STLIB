// -*- C++ -*-

/*!
  \file EikonalAdj1st3.h
  \brief Eikonal equation.  First-order, adjacent scheme.
*/

#if !defined(__hj_EikonalAdj1st3_h__)
#error This file is an implementation detail of the class EikonalAdj1st.
#endif

namespace stlib
{
namespace hj {

//! Eikonal equation.  Adjacent difference scheme.  1st order.
/*!
  \param T is the number type.
*/
template<typename T>
class EikonalAdj1st<3, T> :
   public Eikonal<3, T>,
   public EikonalScheme<3, T> {
private:

   typedef Eikonal<3, T> EquationBase;
   typedef EikonalScheme<3, T> SchemeBase;

   using EquationBase::diff_a1;
   using EquationBase::diff_a1_a1;
   using EquationBase::diff_a1_a1_a1;

   using SchemeBase::_status;
   using SchemeBase::_solution;
   using SchemeBase::_inverseSpeed;

public:

   //! The number type.
   typedef T Number;
   //! A multi-index.
   typedef container::MultiIndexTypes<3>::IndexList IndexList;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   EikonalAdj1st();
   //! Copy constructor not implemented.
   EikonalAdj1st(const EikonalAdj1st&);
   //! Assignment operator not implemented.
   EikonalAdj1st&
   operator=(const EikonalAdj1st&);

public:

   //
   // Constructors
   //

   //! Constructor.
   /*!
     \param solution is the solution array.
     \param status is the status array.
     \param dx is the grid spacing.
   */
   EikonalAdj1st(container::MultiArrayRef<T, 3>& solution,
                 container::MultiArrayRef<Status, 3>& status,
                 const Number dx) :
      EquationBase(dx),
      SchemeBase(solution, status) {}

   //
   // Accessors
   //

   //! Return the radius of the stencil.
   static
   int
   radius() {
      return 1;
   }

   //
   // Manipulators.
   //

   using SchemeBase::getInverseSpeed;

   //
   // Mathematical member functions.
   //

   //! Return a lower bound on correct values.
   Number
   lower_bound(const IndexList& /*i*/, const Number min_unknown) const {
      return min_unknown;
   }


   //! Use the grid point i+di to compute the solution at i.
   Number
   diff_adj(const IndexList& i, const IndexList& d) const;

   //! Use the two specified neighbors to compute the solution at i.
   Number
   diff_adj_adj(const IndexList& i, const IndexList& a, const IndexList& b) const;

   //! Use the three specified neighbors to compute the solution at i.
   Number
   diff_adj_adj_adj(const IndexList& i, const IndexList& a, const IndexList& b,
                    const IndexList& c) const;
};

} // namespace hj
}

#define __hj_EikonalAdj1st3_ipp__
#include "stlib/hj/EikonalAdj1st3.ipp"
#undef __hj_EikonalAdj1st3_ipp__

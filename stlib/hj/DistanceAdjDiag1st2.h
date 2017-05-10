// -*- C++ -*-

/*!
  \file DistanceAdjDiag1st2.h
  \brief Distance equation.  First-order, adjacent-diagonal scheme.
*/

#if !defined(__hj_DistanceAdjDiag1st2_h__)
#error This file is an implementation detail of the class DistanceAdjDiag1st.
#endif

namespace stlib
{
namespace hj {

//! Distance equation.  Adjacent-diagonal difference scheme.  1st order.
/*!
  \param T is the number type.
*/
template<typename T>
class DistanceAdjDiag1st<2, T> :
   public Distance<2, T>,
   public DistanceScheme<2, T> {
private:

   typedef Distance<2, T> EquationBase;
   typedef DistanceScheme<2, T> SchemeBase;

   using EquationBase::diff_a1;
   using EquationBase::diff_d1;
   using EquationBase::diff_a1_d1;

   using EquationBase::_dx_t_sqrt2;
   using EquationBase::_dx_o_sqrt2;

   using SchemeBase::_status;
   using SchemeBase::_solution;

public:

   //! The number type.
   typedef T Number;
   //! A multi-index.
   typedef container::MultiIndexTypes<2>::IndexList IndexList;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   DistanceAdjDiag1st();
   //! Copy constructor not implemented.
   DistanceAdjDiag1st(const DistanceAdjDiag1st&);
   //! Assignment operator not implemented.
   DistanceAdjDiag1st&
   operator=(const DistanceAdjDiag1st&);

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
   DistanceAdjDiag1st(container::MultiArrayRef<T, 2>& solution,
                      container::MultiArrayRef<Status, 2>& status,
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

   //! Return the minimum change from a known solution to a labeled solution when labeling a neighbor.
   Number
   min_delta() const {
      return _dx_o_sqrt2;
   }

   //! Return the maximum change from a known solution to a labeled solution when labeling a neighbor.
   Number
   max_delta() const {
      return _dx_t_sqrt2;
   }

   //
   // Mathematical member functions.
   //

   //! Return a lower bound on correct values.
   Number
   lower_bound(const IndexList& /*i*/, const Number min_unknown) const {
      return min_unknown + _dx_o_sqrt2;
   }

   //! Use the grid point i+d to compute the solution at i.
   Number
   diff_adj(const IndexList& i, const IndexList& d) const;

   //! Use the two specified neighbors to compute the solution at i.
   Number
   diff_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b) const;

   //! Use the grid point i+d to compute the solution at i.
   Number
   diff_diag(const IndexList& i, const IndexList& d) const;

   //! Use the two specified neighbors to compute the solution at i.
   Number
   diff_diag_adj(const IndexList& i, const IndexList& a, const IndexList& b) const;
};

} // namespace hj
}

#define __hj_DistanceAdjDiag1st2_ipp__
#include "stlib/hj/DistanceAdjDiag1st2.ipp"
#undef __hj_DistanceAdjDiag1st2_ipp__

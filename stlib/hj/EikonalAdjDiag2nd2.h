// -*- C++ -*-

/*!
  \file EikonalAdjDiag2nd2.h
  \brief Eikonal equation.  Second-order, adjacent-diagonal scheme.
*/

#if !defined(__hj_EikonalAdjDiag2nd2_h__)
#error This file is an implementation detail of the class EikonalAdjDiag2nd.
#endif

namespace stlib
{
namespace hj {

//! Eikonal equation.  Adjacent-diagonal difference scheme.  2nd order.
/*!
  \param T is the number type.
*/
template<typename T>
class EikonalAdjDiag2nd<2, T> :
   public Eikonal<2, T>,
   public EikonalScheme<2, T> {
private:

   typedef Eikonal<2, T> EquationBase;
   typedef EikonalScheme<2, T> SchemeBase;

   using EquationBase::diff_a1;
   using EquationBase::diff_a2;
   using EquationBase::diff_d1;
   using EquationBase::diff_d2;
   using EquationBase::diff_a1_a1;
   using EquationBase::diff_a1_d1;
   using EquationBase::diff_a1_d2;
   using EquationBase::diff_a2_d1;
   using EquationBase::diff_a2_d2;

   using EquationBase::_dx_t_sqrt2;
   using EquationBase::_dx_o_sqrt2;

   using SchemeBase::_status;
   using SchemeBase::_solution;
   using SchemeBase::_inverseSpeed;

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
   EikonalAdjDiag2nd();
   //! Copy constructor not implemented.
   EikonalAdjDiag2nd(const EikonalAdjDiag2nd&);
   //! Assignment operator not implemented.
   EikonalAdjDiag2nd&
   operator=(const EikonalAdjDiag2nd&);

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
   EikonalAdjDiag2nd(container::MultiArrayRef<T, 2>& solution,
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
      return 2;
   }

   //! Return the minimum change from a known solution to a labeled solution when labeling a neighbor.
   /*!
     This is an expensive function.  It loops over the speed array.
   */
   Number
   min_delta() const {
      return _dx_o_sqrt2 * *std::max_element(_inverseSpeed.begin(),
                                             _inverseSpeed.end());
   }

   //! Return the maximum change from a known solution to a labeled solution when labeling a neighbor.
   /*!
     This is an expensive function.  It loops over the speed array.
   */
   Number
   max_delta() const {
      return _dx_t_sqrt2 * *std::min_element(_inverseSpeed.begin(),
                                             _inverseSpeed.end());
   }

   //
   // Mathematical member functions.
   //

   //! Return a lower bound on correct values.
   Number
   lower_bound(const IndexList& i, const Number min_unknown) const {
      return min_unknown + _dx_o_sqrt2 * _inverseSpeed(i);
   }

   //! Use the grid point i+d to compute the solution at i.
   Number
   diff_adj(const IndexList& i, const IndexList& d) const;

   //! Use the two specified directions to compute the solution at i.
   Number
   diff_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b) const;

   //! Use the direction d to compute the solution at i.
   Number
   diff_diag(const IndexList& i, const IndexList& d) const;

   //! Use the two specified directions to compute the solution at i.
   Number
   diff_diag_adj(const IndexList& i, const IndexList& a, const IndexList& b) const;
};

} // namespace hj
}

#define __hj_EikonalAdjDiag2nd2_ipp__
#include "stlib/hj/EikonalAdjDiag2nd2.ipp"
#undef __hj_EikonalAdjDiag2nd2_ipp__

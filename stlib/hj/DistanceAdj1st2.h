// -*- C++ -*-

/*!
  \file DistanceAdj1st2.h
  \brief Distance equation.  First-order, adjacent scheme.
*/

#if !defined(__hj_DistanceAdj1st2_h__)
#error This file is an implementation detail of the class DistanceAdj1st.
#endif

namespace stlib
{
namespace hj {

//! Distance equation.  Adjacent difference scheme.  1st order.
/*!
  \param T is the number type.
*/
template<typename T>
class DistanceAdj1st<2, T> :
   public Distance<2, T>,
   public DistanceScheme<2, T> {
private:

   typedef Distance<2, T> EquationBase;
   typedef DistanceScheme<2, T> SchemeBase;

   using EquationBase::diff_a1;
   using EquationBase::diff_a1_a1;

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
   DistanceAdj1st();
   //! Copy constructor not implemented.
   DistanceAdj1st(const DistanceAdj1st&);
   //! Assignment operator not implemented.
   DistanceAdj1st&
   operator=(const DistanceAdj1st&);

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
   DistanceAdj1st(container::MultiArrayRef<T, 2>& solution,
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
   diff_adj(const IndexList& i, const IndexList& di) const;

   //! Use the two specified neighbors to compute the solution at i.
   Number
   diff_adj_adj(const IndexList& i, const IndexList& adi, const IndexList& bdi)
      const;

};

} // namespace hj
}

#define __hj_DistanceAdj1st2_ipp__
#include "stlib/hj/DistanceAdj1st2.ipp"
#undef __hj_DistanceAdj1st2_ipp__

// -*- C++ -*-

/*!
  \file GridMCC.h
  \brief An N-D grid that uses marching with a correctness criterion.
*/

#if !defined(__hj_GridMCC_h__)
#define __hj_GridMCC_h__

#include "stlib/hj/Grid.h"

#include "stlib/ads/functor/compare_handle.h"

#include <vector>
#include <algorithm>

namespace stlib
{
namespace hj {

//! The MCC algorithm for solving static H-J equations.
/*!
  This class implements the marching with a correctness criterion for
  solving static Hamilton-Jacobi equations in the \c solve() member
  function.
*/
template<std::size_t N, typename T, class DifferenceScheme>
class GridMCC :
   public Grid<N, T, DifferenceScheme> {
private:

   typedef Grid<N, T, DifferenceScheme> Base;
   //! The multi-index type.
   typedef typename Base::IndexList IndexList;

public:

   //! The number type.
   typedef typename Base::Number Number;

private:

   //! The solution array.
   using Base::_solution;
   //! The finite difference scheme.
   using Base::_scheme;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   GridMCC();
   //! Copy constructor not implemented.
   GridMCC(const GridMCC&);
   //! Assignment operator not implemented.
   GridMCC&
   operator=(const GridMCC&);

public:

   //
   // Constructors
   //

   //! Construct from the solution array and the grid spacing.
   /*!
     \param solution is the solution grid.
     \param dx the spacing between adjacent grid points.
   */
   GridMCC(container::MultiArrayRef<Number, N>& solution, const Number dx) :
      Base(solution, dx) {}

   //! Trivial destructor.
   virtual
   ~GridMCC() {}

   //
   // Mathematical functions.
   //

   //! Solve the Hamilton-Jacobi equation with the MCC method.
   /*!
     Find the solution for all grid points with solution less than or
     equal to \c max_solution.  If \c max_solution is zero, then the solution
     is determined for the entire grid.  The default value of \c max_solution
     is zero.

     This function uses the calls the \c label_neighbors() function
     defined in the finite difference scheme.
   */
   void
   solve(Number max_solution = 0);
};

} // namespace hj
}

#define __hj_GridMCC_ipp__
#include "stlib/hj/GridMCC.ipp"
#undef __hj_GridMCC_ipp__

#endif

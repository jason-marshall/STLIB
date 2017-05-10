// -*- C++ -*-

/*!
  \file GridMCC_CA.h
  \brief An N-D grid that uses the MCC algorithm with a cell array.
*/

#if !defined(__hj_GridMCC_CA_h__)
#define __hj_GridMCC_CA_h__

#include "stlib/hj/Grid.h"
#include "stlib/hj/status.h"

#include "stlib/ads/priority_queue/PriorityQueueCellArray.h"
#include "stlib/ads/functor/compare_handle.h"

#include <vector>

namespace stlib
{
namespace hj {

//! The MCC algorithm with a cell array for solving static H-J equations.
/*!
  This class implements the marching with a correctness criterion for
  solving static Hamilton-Jacobi equations in the \c solve() member
  function.  It uses a cell array to determine which grid points are correct
  at each step.
*/
template<std::size_t N, typename T, class DifferenceScheme>
class GridMCC_CA :
   public Grid<N, T, DifferenceScheme> {
private:

   typedef Grid<N, T, DifferenceScheme> Base;
   //! The index type.
   typedef typename Base::IndexList IndexList;

public:

   //! The number type.
   typedef typename Base::Number Number;
   //! A handle to a value type stored in an array.
   typedef typename Base::handle handle;
   //! A const handle to a value type stored in an array.
   typedef typename Base::const_handle const_handle;

private:

   typedef ads::PriorityQueueCellArray<const_handle> pq_type;
   typedef typename pq_type::container_type container_type;
   typedef typename container_type::const_iterator
   const_handle_const_iterator;

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
   GridMCC_CA();
   //! Copy constructor not implemented.
   GridMCC_CA(const GridMCC_CA&);
   //! Assignment operator not implemented.
   GridMCC_CA&
   operator=(const GridMCC_CA&);

public:

   //
   // Constructors
   //

   //! Construct from the solution array and the grid spacing.
   /*!
     \param solution is the solution grid.
     \param dx the spacing between adjacent grid points.
   */
   GridMCC_CA(container::MultiArrayRef<Number, N>& solution, const Number dx) :
      Base(solution, dx) {}

   //! Trivial destructor.
   virtual
   ~GridMCC_CA() {}

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

#define __hj_GridMCC_CA_ipp__
#include "stlib/hj/GridMCC_CA.ipp"
#undef __hj_GridMCC_CA_ipp__

#endif

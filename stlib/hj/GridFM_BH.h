// -*- C++ -*-

/*!
  \file GridFM_BH.h
  \brief N-D grid.  Fast Marching method.  Binary Heap with static keys.
*/

#if !defined(__hj_GridFM_BH_h__)
#define __hj_GridFM_BH_h__

#include "stlib/hj/Grid.h"
#include "stlib/hj/status.h"

#include "stlib/ads/priority_queue/PriorityQueueBinaryHeapStoreKeys.h"

namespace stlib
{
namespace hj {

//! The fast marching method for solving static H-J equations.
/*!
  This class implements the fast marching method for solving static
  Hamilton-Jacobi equations in the \c solve() member function.  It uses
  a simple priority queue that does not have a decrease_key() operation.
*/
template<std::size_t N, typename T, class DifferenceScheme>
class GridFM_BH :
   public Grid<N, T, DifferenceScheme> {
private:

   typedef Grid<N, T, DifferenceScheme> Base;

protected:

   //! The multi-index type.
   typedef typename Base::IndexList IndexList;

public:

   //! The number type.
   typedef typename Base::Number Number;
   //! A handle to a value type stored in an array.
   typedef typename Base::handle handle;
   //! A const handle to a value type stored in an array.
   typedef typename Base::const_handle const_handle;

protected:

   //! The solution array.
   using Base::_solution;
   //! The finite difference scheme.
   using Base::_scheme;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   GridFM_BH();
   //! Copy constructor not implemented.
   GridFM_BH(const GridFM_BH&);
   //! Assignment operator not implemented.
   GridFM_BH&
   operator=(const GridFM_BH&);

public:

   //
   // Constructors
   //

   //! Construct from the solution array and the grid spacing
   /*!
     \param solution The solution grid.
     \param dx The spacing between adjacent grid points.
   */
   GridFM_BH(container::MultiArrayRef<Number, N>& solution, const Number dx) :
      Base(solution, dx) {}

   //! Trivial destructor.
   virtual
   ~GridFM_BH() {}

   //
   // Mathematical functions.
   //

   //! Solve the Hamilton-Jacobi equation with the FM method.
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

#define __hj_GridFM_BH_ipp__
#include "stlib/hj/GridFM_BH.ipp"
#undef __hj_GridFM_BH_ipp__

#endif

// -*- C++ -*-

/*!
  \file GridSort.h
  \brief N-D grid.  Sort method.  Models an ideal method.
*/

#if !defined(__hj_GridSort_h__)
#define __hj_GridSort_h__

#include "stlib/hj/GridFM_BH.h"

#include "stlib/ads/functor/compare_handle.h"

namespace stlib
{
namespace hj {

//! The sort algorithm for "solving" static H-J equations.
/*!
  This class models an ideal method for solving static
  Hamilton-Jacobi equations in the \c solve() member function.
  To use this class:
  - Set the initial condition.
  - Call pre_solve() to sort the grid points.
  - Set the initial condition again.
  - Call the solve() function, which models an ideal method.
*/
template<std::size_t N, typename T, class DifferenceScheme>
class GridSort :
   public GridFM_BH<N, T, DifferenceScheme> {
private:

   typedef GridFM_BH<N, T, DifferenceScheme> Base;
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

   typedef std::vector<const_handle> const_handle_container;
   typedef typename const_handle_container::const_iterator
   const_handle_const_iterator;

private:

   //! The solution array.
   using Base::_solution;
   //! The finite difference scheme.
   using Base::_scheme;

private:

   //
   // Member data.
   //

   // The sorted grid points.
   std::vector<const_handle> _sorted;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   GridSort();
   //! Copy constructor not implemented.
   GridSort(const GridSort&);
   //! Assignment operator not implemented.
   GridSort&
   operator=(const GridSort&);

public:

   //
   // Constructors
   //

   //! Construct from the solution array and the grid spacing.
   /*!
     \param solution is the solution grid.
     \param dx the spacing between adjacent grid points.
   */
   GridSort(container::MultiArrayRef<Number, N>& solution, const Number dx) :
      Base(solution, dx),
      _sorted() {
      // Reserve space for the array of handles into the solution array.
      _sorted.reserve(_solution.size());
      // Set the handles.
      for (const_handle i = _solution.begin(); i != _solution.end(); ++i) {
         _sorted.push_back(i);
      }
   }

   //! Trivial destructor.
   virtual
   ~GridSort() {}

   //
   // Mathematical functions.
   //

   //! Sort the grid points.
   /*!
     Find the solution for all grid points with solution less than or
     equal to \c max_solution.  If \c max_solution is zero, then the solution
     is determined for the entire grid.  The default value of \c max_solution
     is zero.

     Solve the problem with the fast marching method.  Then sort the
     grid points.
   */
   void
   pre_solve(Number max_solution = 0);

   //! Solve the Hamilton-Jacobi equation with the an ideal method.
   /*!
     Note that the \c max_solution argument is not used.  It is used in
     pre_solve.

     This function uses the calls the \c label_neighbors() function
     defined in the finite difference scheme.
   */
   void
   solve(Number max_solution = 0);
};

} // namespace hj
}

#define __hj_GridSort_ipp__
#include "stlib/hj/GridSort.ipp"
#undef __hj_GridSort_ipp__

#endif

// -*- C++ -*-

#if !defined(__hj_GridSort_ipp__)
#error This file is an implementation detail of the class GridSort.
#endif

namespace stlib
{
namespace hj {

template<std::size_t N, typename T, class DifferenceScheme>
inline
void
GridSort<N, T, DifferenceScheme>::
pre_solve(const Number max_solution) {
   // Solve the problem with the fast marching method.
   Base::solve(max_solution);
   // Sort the grid points in the solution.
   std::sort(_sorted.begin(), _sorted.end(),
             ads::LessByHandle<const_handle>());
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
GridSort<N, T, DifferenceScheme>::
solve(const Number max_solution) {
   IndexList i;
   int dummy = 0;

   // If we are going to solve for all grid points.
   if (max_solution == 0) {
      // All vertices are known when there are no labeled vertices left.
      // Loop while there are labeled vertices left.
      for (const_handle_const_iterator iter = _sorted.begin();
            iter != _sorted.end(); ++iter) {
         // Convert the handle to indices.
         _solution.indexList(*iter - _solution.data(), &i);
         // Label the neighbors.
         _scheme.label_neighbors(dummy, i);
      }
   }
   // Else we solve for the grid points around the initial condition.
   else {
      // Loop while there are labeled vertices left and while the solution
      // is less than or equal to max_solution.
      for (const_handle_const_iterator iter = _sorted.begin();
            iter != _sorted.end() &&** iter <= max_solution; ++iter) {
         // Convert the handle to indices.
         _solution.indexList(*iter - _solution.data(), &i);
         // Label the neighbors.
         _scheme.label_neighbors(dummy, i);
      }
   }
}

} // namespace hj
}

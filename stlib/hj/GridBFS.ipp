// -*- C++ -*-

#if !defined(__hj_GridBFS_ipp__)
#error This file is an implementation detail of the class GridBFS.
#endif

namespace stlib
{
namespace hj {

template<std::size_t N, typename T, class DifferenceScheme>
inline
void
GridBFS<N, T, DifferenceScheme>::
solve(const Number max_solution) {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   IndexList i;

   // The container of labeled unknown grid points.
   std::deque<Number*> labeled;

   // Label the neighbors of known grid points.
   {
      const Iterator end = Iterator::end(_solution.range());
      for (Iterator i = Iterator::begin(_solution.range()); i != end; ++i) {
         if (_scheme.is_initial(*i)) {
            _scheme.label_neighbors(labeled, *i);
         }
      }
   }

   // If we are going to solve for all grid points.
   if (max_solution == 0) {
      // All vertices are known when there are no labeled vertices left.
      // Loop while there are labeled vertices left.
      while (! labeled.empty()) {
         // Get a grid point.
         _solution.indexList(labeled.front() - _solution.data(), &i);
         labeled.pop_front();
         // Label the adjacent neighbors.
         _scheme.label_neighbors(labeled, i);
      }
   }
   // Else we solve for the grid points around the initial condition.
   else {
      // Loop while there are labeled vertices left and while the solution
      // is less than or equal to max_solution.
      Number soln = max_solution;
      while (! labeled.empty() && soln <= max_solution) {
         // Get a grid point.
         _solution.indexList(labeled.front() - _solution.data(), &i);
         soln = _solution(i);
         labeled.pop_front();
         // Label the adjacent neighbors.
         _scheme.label_neighbors(labeled, i);
      }
   }
}

} // namespace hj
}

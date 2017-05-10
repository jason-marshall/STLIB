// -*- C++ -*-

#if !defined(__hj_GridMCC_CA_ipp__)
#error This file is an implementation detail of the class GridMCC_CA.
#endif

namespace stlib
{
namespace hj {

template<std::size_t N, typename T, class DifferenceScheme>
inline
void
GridMCC_CA<N, T, DifferenceScheme>::
solve(const Number max_solution) {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   IndexList i;

   // Find the known grid points.
   container_type initial;
   {
      const Iterator end = Iterator::end(_solution.range());
      for (Iterator i = Iterator::begin(_solution.range()); i != end; ++i) {
         if (_scheme.is_initial(*i)) {
            initial.push_back(&_solution(*i));
         }
      }
   }

   // Determine the minimum solution in the initial condition.
   Number min_initial =
      **std::min_element(initial.begin(), initial.end(),
                         ads::LessByHandle<const_handle>());

   // The priority queue of labeled grid points.
   pq_type labeled(min_initial, _scheme.equation().min_delta(),
                   _scheme.equation().max_delta());

   // Label the neighbors of known grid points.
   for (typename container_type::const_iterator iter = initial.begin();
         iter != initial.end(); ++iter) {
      // Convert the handle to indices.
      _solution.indexList(*iter - _solution.data(), &i);
      // Label the neighbors.
      _scheme.label_neighbors(labeled, i);
   }

   // Free the memory for the initial condition.
   {
      container_type tmp;
      tmp.swap(initial);
   }

   const_handle_const_iterator iter, end;

   // If we are going to solve for all grid points.
   if (max_solution == 0) {
      // All vertices are known when there are no labeled vertices left.
      // Loop while there are labeled vertices left.
      while (! labeled.empty()) {
         end = labeled.top().end();
         // Loop over the cell.
         for (iter = labeled.top().begin(); iter != end; ++iter) {
            // Convert the handle to indices.
            _solution.indexList(*iter - _solution.data(), &i);
            // If the grid point has not already been set.
            if (_scheme.status()(i) == LABELED) {
               // Label the neighbors.
               _scheme.label_neighbors(labeled, i);
            }
         }
         // Remove the cell at the top of the priority queue.
         labeled.pop();
      }
   }
   // Else we solve for the grid points around the initial condition.
   else {
      // Loop while there are labeled vertices left and while the solution
      // is less than or equal to max_solution.
      while (! labeled.empty() && labeled.lower_bound() <= max_solution) {
         end = labeled.top().end();
         // Loop over the cell.
         for (iter = labeled.top().begin(); iter != end; ++iter) {
            // Convert the handle to indices.
            _solution.indexList(*iter - _solution.data(), &i);
            // If the grid point has not already been set.
            if (_scheme.status()(i) == LABELED) {
               // Label the neighbors.
               _scheme.label_neighbors(labeled, i);
            }
         }
         // Remove the cell at the top of the priority queue.
         labeled.pop();
      }
   }
}

} // namespace hj
}

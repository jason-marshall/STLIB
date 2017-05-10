// -*- C++ -*-

#if !defined(__hj_Grid_ipp__)
#error This file is an implementation detail of the class Grid.
#endif

namespace stlib
{
namespace hj {

//
// Constructors
//


template<std::size_t N, typename T, class DifferenceScheme>
inline
Grid<N, T, DifferenceScheme>::
Grid(container::MultiArrayRef<Number, N>& solution, const Number dx,
     const bool is_concurrent) :
   _solution(solution),
   // For the concurrent algorithm the solution grid has ghost points
   // made by enlarging the grid by the radius of the finite difference
   // scheme.
   _ranges(_solution.extents() +
           std::size_t(is_concurrent ? 2 * DifferenceScheme::radius() : 0),
           _solution.bases() - Index(is_concurrent ?
                                     DifferenceScheme::radius() : 0)),
   _dx(dx),
   _scheme(_ranges, _solution, _dx, is_concurrent) {}


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
initialize() {
   // Set the solution to infinity.
   std::fill(_solution.begin(), _solution.end(),
             std::numeric_limits<Number>::max());

   // Initialize the difference scheme.
   _scheme.initialize();
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
std::size_t
Grid<N, T, DifferenceScheme>::
set_unsigned_initial_condition() {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   // Initialize the difference scheme.
   _scheme.initialize();

   // Loop over the solution array.
   std::size_t count = 0;
   const Iterator end = Iterator::end(_solution.range());
   for (Iterator i = Iterator::begin(_solution.range()); i != end; ++i) {
      T& s = _solution(*i);
      // If the solution is known.
      if (s != std::numeric_limits<Number>::max()) {
         // If the grid point has an unkown neighbor.
         if (_scheme.has_unknown_neighbor(*i)) {
            // Set the status of the solution as known in the initial condition.
            _scheme.set_initial(*i);
            ++count;
         }
         else {
            // Set the status to KNOWN.
            _scheme.set_known(*i);
         }
      }
   }
   return count;
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
std::size_t
Grid<N, T, DifferenceScheme>::
set_negative_initial_condition() {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   // Initialize the difference scheme.
   _scheme.initialize();

   // Loop over the solution array.
   std::size_t initial = 0;
   const Iterator end = Iterator::end(_solution.range());
   for (Iterator i = Iterator::begin(_solution.range()); i != end; ++i) {
      T& s = _solution(*i);
      // If the solution is finite.
      if (s != std::numeric_limits<Number>::max()) {
         // If the solution is non-positive.
         if (s <= 0 && _scheme.has_unknown_neighbor(*i)) {
            // Set the status as known in the initial condition.
            _scheme.set_initial(*i);
            ++initial;
         }
         else {
            // Set the status to KNOWN.
            _scheme.set_known(*i);
         }
         // Reverse the sign of the solution.
         s = - s;
      }
   }
   return initial;
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
std::size_t
Grid<N, T, DifferenceScheme>::
set_positive_initial_condition() {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   // Loop over the solution array.
   std::size_t initial = 0;
   const Iterator end = Iterator::end(_solution.range());
   for (Iterator i = Iterator::begin(_solution.range()); i != end; ++i) {
      // If the solution is known.
      if (_scheme.is_known(*i)) {
         T& s = _solution(*i);
#ifdef STLIB_DEBUG
         assert(s != std::numeric_limits<Number>::max());
#endif
         s = - s;
         if (s >= 0 && _scheme.has_unknown_neighbor(*i)) {
            _scheme.set_initial(*i);
            ++initial;
         }
      }
   }
   return initial;
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
set_initial(const IndexList& i, const Number value) {
   _solution(i) = value;
   _scheme.set_initial(i);
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
add_source(const std::array<Number, 2>& x) {
   static_assert(N == 2, "The dimension must be 2.");

   assert(_scheme.equation().radius() == 1 ||
          _scheme.equation().radius() == 2);

   assert(_solution.bases()[0] <= x[0] &&
          x[0] <= _solution.bases()[0] + _solution.extents()[0] - 1 &&
          _solution.bases()[1] <= x[1] &&
          x[1] <= _solution.bases()[1] + _solution.extents()[1] - 1);


   const Index radius = _scheme.equation().radius() - 1;
   const Number eps = std::sqrt(std::numeric_limits<Number>::epsilon());
   // Floor.
   const Index i_start = Index(x[0] - eps) - radius;
   const Index j_start = Index(x[1] - eps) - radius;
   // Ceiling.
   const Index i_stop = Index(x[0] + eps) + 1 + radius;
   const Index j_stop = Index(x[1] + eps) + 1 + radius;
   IndexList i;
   for (i[1] = j_start; i[1] <= j_stop; ++i[1]) {
      for (i[0] = i_start; i[0] <= i_stop; ++i[0]) {
         if (indices_in_grid(i)) {
            set_initial(i, std::min(_solution(i), index_distance(i, x)));
         }
      }
   }
}


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
add_source(const std::array<Number, 3>& x) {
   static_assert(N == 3, "The dimension must be 3.");

   assert(_scheme.equation().radius() == 1 ||
          _scheme.equation().radius() == 2);

   assert(_solution.bases()[0] <= x[0] &&
          x[0] <= _solution.bases()[0] + _solution.extents()[0] - 1 &&
          _solution.bases()[1] <= x[1] &&
          x[1] <= _solution.bases()[1] + _solution.extents()[1] - 1 &&
          _solution.bases()[2] <= x[2] &&
          x[1] <= _solution.bases()[2] + _solution.extents()[2] - 1);


   const Index radius = _scheme.equation().radius() - 1;
   const Number eps = std::sqrt(std::numeric_limits<Number>::epsilon());
   // Floor.
   const Index i_start = Index(x[0] - eps) - radius;
   const Index j_start = Index(x[1] - eps) - radius;
   const Index k_start = Index(x[2] - eps) - radius;
   // Ceiling.
   const Index i_stop = Index(x[0] + eps) + 1 + radius;
   const Index j_stop = Index(x[1] + eps) + 1 + radius;
   const Index k_stop = Index(x[2] + eps) + 1 + radius;
   IndexList i;
   for (i[2] = k_start; i[2] <= k_stop; ++i[2]) {
      for (i[1] = j_start; i[1] <= j_stop; ++i[1]) {
         for (i[0] = i_start; i[0] <= i_stop; ++i[0]) {
            if (indices_in_grid(i)) {
               set_initial(i, std::min(_solution(i), index_distance(i, x)));
            }
         }
      }
   }
}


//
// I/O
//


template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
print_statistics(std::ostream& out) const {
   // Print statistics for the status array.
   _scheme.print_statistics(out);

   T min_known = std::numeric_limits<T>::max();
   T max_known = -std::numeric_limits<T>::max();
   T s;
   std::size_t num_known = 0;
   std::size_t num_positive = 0;
   std::size_t num_nonpositive = 0;
   const std::size_t size = _solution.size();
   for (std::size_t n = 0; n != size; ++n) {
      s = _solution[n];
      if (s != std::numeric_limits<T>::max()) {
         ++num_known;

         if (s > 0) {
            ++num_positive;
         }
         else {
            ++num_nonpositive;
         }

         if (s < min_known) {
            min_known = s;
         }
         if (s > max_known) {
            max_known = s;
         }
      }
   }

   out << "Solution array size = " << size << "\n"
       << "  Number known =        " << num_known << "\n"
       << "  Number positive =     " << num_positive << "\n"
       << "  Number non-positive = " << num_nonpositive << "\n"
       << "  Minimum known =       " << min_known << "\n"
       << "  Maximum known =       " << max_known << "\n";
}


template<typename T>
inline
void
print_solution_array(std::ostream& out,
                     const container::MultiArrayConstRef<T, 2>& solution) {
   typedef container::MultiArrayConstRef<T, 2> MultiArray;
   typedef typename MultiArray::IndexList IndexList;
   typedef typename MultiArray::Index Index;

   // Write the solution.
   const IndexList upper = solution.bases() +
      ext::convert_array<Index>(solution.extents());
   out << "Solution:" << '\n';
   for (Index j = upper[1] - 1; j >= solution.bases()[1]; --j) {
      for (Index i = solution.bases()[0]; i < upper[0]; ++i) {
         out << solution(i, j) << " ";
      }
      out << '\n';
   }
}


template<typename T>
inline
void
print_solution_array(std::ostream& out,
                     const container::MultiArrayConstRef<T, 3>& solution) {
   typedef container::MultiArrayConstRef<T, 3> MultiArray;
   typedef typename MultiArray::IndexList IndexList;
   typedef typename MultiArray::Index Index;

   // Write the solution.
   const IndexList upper = solution.bases() +
      ext::convert_array<Index>(solution.extents());
   out << "Solution:" << '\n';
   for (Index k = upper[2] - 1; k >= solution.bases()[2]; --k) {
      for (Index j = upper[1] - 1; j >= solution.bases()[1]; --j) {
         for (Index i = solution.bases()[0]; i < upper[0]; ++i) {
            out << solution(i, j, k) << " ";
         }
         out << '\n';
      }
      out << '\n';
   }
}

template<std::size_t N, typename T, class DifferenceScheme>
inline
void
Grid<N, T, DifferenceScheme>::
put(std::ostream& out) const {
   // Write the solution array.
   print_solution_array(out, _solution);
   // Write information from the difference scheme.
   out << _scheme;
}

//
// File I/O
//

template<std::size_t N, typename T, class DifferenceScheme>
inline
std::ostream&
operator<<(std::ostream& out, const Grid<N, T, DifferenceScheme>& x) {
   x.put(out);
   return out;
}

} // namespace hj
}

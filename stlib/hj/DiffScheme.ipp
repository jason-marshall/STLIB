// -*- C++ -*-

#if !defined(__hj_DiffScheme_ipp__)
#error This file is an implementation detail of the class DiffScheme.
#endif

namespace stlib
{
namespace hj {


template<std::size_t N, typename T, class Equation>
inline
DiffScheme<N, T, Equation>::
DiffScheme(const Range& index_ranges,
           container::MultiArrayRef<Number, N>& solution,
           const Number dx, const bool is_concurrent) :
   _index_ranges(index_ranges),
   _solution(solution),
   // For the sequential algorithm, the status array is larger than the
   // solution array.  It has an extra layer that is one grid point thick.
   // This is because grid points on the edge of the solution array will
   // call label_neighbors.  For the concurrent algorithm, the status array
   // needs to have two extra grid points around the original solution array.
   // (By "original" I mean the solution array without the ghost points.
   // This is because the GHOST_ADJACENT points will call label_neighbors.
   _status(_index_ranges.extents() + std::size_t(is_concurrent ? 4 : 2),
           _index_ranges.bases() - std::ptrdiff_t(is_concurrent ? 2 : 1)),
   _equation(solution, _status, dx) {
}


template<std::size_t N, typename T, class Equation>
inline
void
DiffScheme<N, T, Equation>::
initialize() {
   typedef container::MultiIndexRangeIterator<N> Iterator;

   //
   // Set the status of the grid points.
   //

   // First make everything a void point.
   std::fill(_status.begin(), _status.end(), VOID);

   // Then mark the interior points an unlabeled.
   // Loop over the indices.
   const Iterator end = Iterator::end(_index_ranges);
   for (Iterator begin = Iterator::begin(_index_ranges); begin != end;
        ++begin) {
      _status(*begin) = UNLABELED;
   }
}


template<std::size_t N, typename T, class Equation>
template<class Container>
inline
void
DiffScheme<N, T, Equation>::
label(Container& unlabeled_neighbors, const IndexList& i, const Number value) {
   Status& stat = _status(i);
   Number& soln = _solution(i);

#ifdef STLIB_DEBUG
   assert(stat == UNLABELED || stat == LABELED);
#endif

   if (stat == UNLABELED) {
      stat = LABELED;
      soln = value;
      unlabeled_neighbors.push_back(&soln);
   }
   else if (value < soln) {
      soln = value;
   }
}


template<std::size_t N, typename T, class Equation>
template<typename HandleType>
inline
void
DiffScheme<N, T, Equation>::
label(ads::PriorityQueueBinaryHeapArray<HandleType>& labeled,
      const IndexList& i, const Number value) {
   Status& stat = _status(i);
   Number& soln = _solution(i);

#ifdef STLIB_DEBUG
   assert(stat == UNLABELED || stat == LABELED);
#endif

   if (stat == UNLABELED) {
      stat = LABELED;
      soln = value;
      labeled.push(&soln);
   }
   else if (value < soln) {
      soln = value;
      labeled.decrease(&soln);
   }
}


template<std::size_t N, typename T, class Equation>
template<typename HandleType>
inline
void
DiffScheme<N, T, Equation>::
label(ads::PriorityQueueBinaryHeapStoreKeys<HandleType>& labeled,
      const IndexList& i, const Number value) {
   // CONTINUE: REMOVE
   //std::cerr << "label(labeled, " << i << ", " << value << ")\n";

   Status& stat = _status(i);
   Number& soln = _solution(i);

#ifdef STLIB_DEBUG
   assert(stat == UNLABELED || stat == LABELED);
#endif

   if (stat == UNLABELED) {
      stat = LABELED;
      soln = value;
      labeled.push(&soln);
   }
   else if (value < soln) {
      soln = value;
      labeled.push(&soln);
   }
   //std::cerr << "Done label().\n";
}


template<std::size_t N, typename T, class Equation>
template<typename HandleType>
inline
void
DiffScheme<N, T, Equation>::
label(ads::PriorityQueueCellArray<HandleType>& labeled,
      const IndexList& i, const Number value) {
   Status& stat = _status(i);
   Number& soln = _solution(i);

#ifdef STLIB_DEBUG
   assert(stat == UNLABELED || stat == LABELED);
#endif

   if (stat == UNLABELED) {
      stat = LABELED;
      soln = value;
      labeled.push(&soln);
   }
   else if (value < soln) {
      soln = value;
      labeled.push(&soln);
   }
}


template<std::size_t N, typename T, class Equation>
inline
void
DiffScheme<N, T, Equation>::
label(int, const IndexList& i, const Number value) {
   Status& stat = _status(i);
   Number& soln = _solution(i);

#ifdef STLIB_DEBUG
   assert(stat == UNLABELED || stat == LABELED);
#endif

   if (stat == UNLABELED) {
      stat = LABELED;
      soln = value;
   }
   else if (value < soln) {
      soln = value;
   }
}


//
// File I/O
//


template<std::size_t N, typename T, class Equation>
inline
void
DiffScheme<N, T, Equation>::
print_statistics(std::ostream& out) const {
   std::size_t num_known = 0;
   std::size_t num_labeled = 0;
   std::size_t num_unlabeled = 0;
   std::size_t num_void = 0;
   std::size_t num_initial = 0;
   Status s;
   // Loop over the status array.
   for (std::size_t n = 0; n != _status.size(); ++n) {
      s = _status[n];
      if (s == KNOWN) {
         ++num_known;
      }
      else if (s == LABELED) {
         ++num_labeled;
      }
      else if (s == UNLABELED) {
         ++num_unlabeled;
      }
      else if (s == VOID) {
         ++num_void;
      }
      else if (s == INITIAL) {
         ++num_initial;
      }
      else {
         assert(false);
      }
   }

   out << "Status array size = " << _status.size() << "\n"
       << "  Number KNOWN =     " << num_known << "\n"
       << "  Number LABELED =   " << num_labeled << "\n"
       << "  Number UNLABELED = " << num_unlabeled << "\n"
       << "  Number VOID =      " << num_void << "\n"
       << "  Number INITIAL =   " << num_initial << "\n";
}


inline
char
status_character(const Status s) {
   if (s == KNOWN) {
      return 'K';
   }
   else if (s == LABELED) {
      return 'L';
   }
   else if (s == UNLABELED) {
      return 'U';
   }
   else if (s == VOID) {
      return 'V';
   }
   else if (s == INITIAL) {
      return 'I';
   }
   else {
      assert(false);
   }
   return ' ';
}


inline
void
print_status_array(std::ostream& out,
                   const container::MultiArrayConstRef<Status, 2>& status) {
   typedef container::MultiArrayConstRef<Status, 2>::Index Index;
   typedef container::MultiArrayConstRef<Status, 2>::IndexList IndexList;
   out << "Status:" << '\n';
   const IndexList upper = status.bases() +
      ext::convert_array<Index>(status.extents());
   for (Index j = upper[1] - 1; j >= status.bases()[1]; --j) {
      for (Index i = status.bases()[0]; i < upper[0]; ++i) {
         out << status_character(status(i, j)) << ' ';
      }
      out << '\n';
   }
}


inline
void
print_status_array(std::ostream& out,
                   const container::MultiArrayConstRef<Status, 3>& status) {
   typedef container::MultiArrayConstRef<Status, 3>::Index Index;
   typedef container::MultiArrayConstRef<Status, 3>::IndexList IndexList;

   out << "Status:" << '\n';
   const IndexList upper = status.bases() +
      ext::convert_array<Index>(status.extents());
   for (Index k = upper[2] - 1; k >= status.bases()[2]; --k) {
      for (Index j = upper[1] - 1; j >= status.bases()[1]; --j) {
         for (Index i = status.bases()[0]; i < upper[0]; ++i) {
            out << status_character(status(i, j, k)) << ' ';
         }
         out << '\n';
      }
      out << '\n';
   }
}


template<std::size_t N, typename T, class Equation>
inline
void
DiffScheme<N, T, Equation>::
put(std::ostream& out) const {
   print_status_array(out, _status);
   out << _equation;
}


template<std::size_t N, typename T, class Equation>
std::ostream&
operator<<(std::ostream& out, const DiffScheme<N, T, Equation>& x) {
   x.put(out);
   return out;
}


} // namespace hj
}

// -*- C++ -*-

#if !defined(__hj_DiffSchemeAdj2_ipp__)
#error This file is an implementation detail of the class DiffSchemeAdj.
#endif

namespace stlib
{
namespace hj {

template<typename T, class Equation>
inline
bool
DiffSchemeAdj<2, T, Equation>::
has_unknown_neighbor(const IndexList& i) const {
   if ((_status(i[0] - 1, i[1]) == UNLABELED &&
        _solution(i[0] - 1, i[1]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1]) == UNLABELED &&
        _solution(i[0] + 1, i[1]) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] - 1) == UNLABELED &&
        _solution(i[0], i[1] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] + 1) == UNLABELED &&
        _solution(i[0], i[1] + 1) == std::numeric_limits<T>::max())) {
      return true;
   }
   return false;
}

template<typename T, class Equation>
template<class Container>
inline
void
DiffSchemeAdj<2, T, Equation>::
label_neighbors(Container& labeled, IndexList i) {
   // This grid point is now KNOWN.
   _status(i) = KNOWN;

   //
   // Label in the adjacent directions.
   //
   IndexList di;

   // (-1,0)
   --i[0];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = 0;
      label(labeled, i, diff(i, di));
   }
   ++i[0];

   // (1,0)
   ++i[0];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 0;
      label(labeled, i, diff(i, di));
   }
   --i[0];

   // (0,-1)
   --i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 1;
      label(labeled, i, diff(i, di));
   }
   ++i[1];

   // (0,1)
   ++i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = -1;
      label(labeled, i, diff(i, di));
   }
   --i[1];
}


template<typename T, class Equation>
inline
typename DiffSchemeAdj<2, T, Equation>::Number
DiffSchemeAdj<2, T, Equation>::
diff(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   return std::min(_equation.diff_adj(i, di), diff_adj_adj(i, di));
}


template<typename T, class Equation>
inline
typename DiffSchemeAdj<2, T, Equation>::Number
DiffSchemeAdj<2, T, Equation>::
diff_adj_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   IndexList dj = {{di[1], di[0]}};
   const Number a = _equation.diff_adj_adj(i, di, dj);
   dj[0] = - di[1];
   dj[1] = - di[0];
   const Number b = _equation.diff_adj_adj(i, di, dj);
   return std::min(a, b);
}

} // namespace hj
}

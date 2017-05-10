// -*- C++ -*-

#if !defined(__hj_DiffSchemeAdj3_ipp__)
#error This file is an implementation detail of the class DiffSchemeAdj.
#endif

namespace stlib
{
namespace hj {

template<typename T, class Equation>
inline
bool
DiffSchemeAdj<3, T, Equation>::
has_unknown_neighbor(const IndexList& i) const {
   if ((_status(i[0] - 1, i[1], i[2]) == UNLABELED &&
        _solution(i[0] - 1, i[1], i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1], i[2]) == UNLABELED &&
        _solution(i[0] + 1, i[1], i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] - 1, i[2]) == UNLABELED &&
        _solution(i[0], i[1] - 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] + 1, i[2]) == UNLABELED &&
        _solution(i[0], i[1] + 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1], i[2] - 1) == UNLABELED &&
        _solution(i[0], i[1], i[2] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1], i[2] + 1) == UNLABELED &&
        _solution(i[0], i[1], i[2] + 1) == std::numeric_limits<T>::max())) {
      return true;
   }

   return false;
}


template<typename T, class Equation>
template<class Container>
inline
void
DiffSchemeAdj<3, T, Equation>::
label_neighbors(Container& labeled, IndexList i) {
   // This grid point is now KNOWN.
   _status(i) = KNOWN;

   //
   // Label in the adjacent directions.
   //
   IndexList di;

   // (-1,0,0)
   --i[0];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = 0;
      di[2] = 0;
      label(labeled, i, diff(i, di));
   }
   ++i[0];

   // (1,0,0)
   ++i[0];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 0;
      di[2] = 0;
      label(labeled, i, diff(i, di));
   }
   --i[0];

   // (0,-1,0)
   --i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 1;
      di[2] = 0;
      label(labeled, i, diff(i, di));
   }
   ++i[1];

   // (0,1,0)
   ++i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = -1;
      di[2] = 0;
      label(labeled, i, diff(i, di));
   }
   --i[1];

   // (0,0,-1)
   --i[2];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 0;
      di[2] = 1;
      label(labeled, i, diff(i, di));
   }
   ++i[2];

   // (0,0,1)
   ++i[2];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 0;
      di[2] = -1;
      label(labeled, i, diff(i, di));
   }
   --i[2];
}


template<typename T, class Equation>
inline
typename DiffSchemeAdj<3, T, Equation>::Number
DiffSchemeAdj<3, T, Equation>::
diff(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   return ads::min(_equation.diff_adj(i, di),
                   diff_adj_adj(i, di),
                   diff_adj_adj_adj(i, di));
}


template<typename T, class Equation>
inline
typename DiffSchemeAdj<3, T, Equation>::Number
DiffSchemeAdj<3, T, Equation>::
diff_adj_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   IndexList dj = {{di[2], di[0], di[1]}};
   const Number x1 = _equation.diff_adj_adj(i, di, dj);

   dj[0] = - di[2];
   dj[1] = - di[0];
   dj[2] = - di[1];
   const Number x2 = _equation.diff_adj_adj(i, di, dj);

   dj[0] = di[1];
   dj[1] = di[2];
   dj[2] = di[0];
   const Number x3 = _equation.diff_adj_adj(i, di, dj);

   dj[0] = - di[1];
   dj[1] = - di[2];
   dj[2] = - di[0];
   const Number x4 = _equation.diff_adj_adj(i, di, dj);

   return ads::min(x1, x2, x3, x4);
}


template<typename T, class Equation>
inline
typename DiffSchemeAdj<3, T, Equation>::Number
DiffSchemeAdj<3, T, Equation>::
diff_adj_adj_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   IndexList dj = {{di[2], di[0], di[1]}};
   IndexList dk = {{di[1], di[2], di[0]}};
   const Number x1 = _equation.diff_adj_adj_adj(i, di, dj, dk);

   dj[0] = di[2];
   dj[1] = di[0];
   dj[2] = di[1];
   dk[0] = - di[1];
   dk[1] = - di[2];
   dk[2] = - di[0];
   const Number x2 = _equation.diff_adj_adj_adj(i, di, dj, dk);

   dj[0] = - di[2];
   dj[1] = - di[0];
   dj[2] = - di[1];
   dk[0] = di[1];
   dk[1] = di[2];
   dk[2] = di[0];
   const Number x3 = _equation.diff_adj_adj_adj(i, di, dj, dk);

   dj[0] = - di[2];
   dj[1] = - di[0];
   dj[2] = - di[1];
   dk[0] = - di[1];
   dk[1] = - di[2];
   dk[2] = - di[0];
   const Number x4 = _equation.diff_adj_adj_adj(i, di, dj, dk);

   return ads::min(x1, x2, x3, x4);
}

} // namespace hj
}

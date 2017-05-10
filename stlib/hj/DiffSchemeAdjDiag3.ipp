// -*- C++ -*-

#if !defined(__DiffSchemeAdjDiag3_ipp__)
#error This file is an implementation detail of the class DiffSchemeAdjDiag
#endif

namespace stlib
{
namespace hj {

template<typename T, class Equation>
inline
bool
DiffSchemeAdjDiag<3, T, Equation>::
has_unknown_neighbor(const IndexList& i) const {
   if (// Adjacent
       (_status(i[0] - 1, i[1], i[2]) == UNLABELED &&
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
        _solution(i[0], i[1], i[2] + 1) == std::numeric_limits<T>::max()) ||
      // Diagonal
       (_status(i[0] - 1, i[1] - 1, i[2]) == UNLABELED &&
        _solution(i[0] - 1, i[1] - 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] - 1, i[1] + 1, i[2]) == UNLABELED &&
        _solution(i[0] - 1, i[1] + 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1] - 1, i[2]) == UNLABELED &&
        _solution(i[0] + 1, i[1] - 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1] + 1, i[2]) == UNLABELED &&
        _solution(i[0] + 1, i[1] + 1, i[2]) == std::numeric_limits<T>::max()) ||
       (_status(i[0] - 1, i[1], i[2] - 1) == UNLABELED &&
        _solution(i[0] - 1, i[1], i[2] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0] - 1, i[1], i[2] + 1) == UNLABELED &&
        _solution(i[0] - 1, i[1], i[2] + 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1], i[2] - 1) == UNLABELED &&
        _solution(i[0] + 1, i[1], i[2] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0] + 1, i[1], i[2] + 1) == UNLABELED &&
        _solution(i[0] + 1, i[1], i[2] + 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] - 1, i[2] - 1) == UNLABELED &&
        _solution(i[0], i[1] - 1, i[2] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] - 1, i[2] + 1) == UNLABELED &&
        _solution(i[0], i[1] - 1, i[2] + 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] + 1, i[2] - 1) == UNLABELED &&
        _solution(i[0], i[1] + 1, i[2] - 1) == std::numeric_limits<T>::max()) ||
       (_status(i[0], i[1] + 1, i[2] + 1) == UNLABELED &&
        _solution(i[0], i[1] + 1, i[2] + 1) == std::numeric_limits<T>::max())) {
      return true;
   }

   return false;
}


template<typename T, class Equation>
template<class Container>
inline
void
DiffSchemeAdjDiag<3, T, Equation>::
label_neighbors(Container& labeled, IndexList i) {
   // CONTINUE: REMOVE
   //std::cerr << "label_neighbors(labeled, " << i << ")\n";

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
      label(labeled, i, diff_using_adj(i, di));
   }
   ++i[0];

   // (1,0,0)
   ++i[0];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 0;
      di[2] = 0;
      label(labeled, i, diff_using_adj(i, di));
   }
   --i[0];

   // (0,-1,0)
   --i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 1;
      di[2] = 0;
      label(labeled, i, diff_using_adj(i, di));
   }
   ++i[1];

   // (0,1,0)
   ++i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = -1;
      di[2] = 0;
      label(labeled, i, diff_using_adj(i, di));
   }
   --i[1];

   // (0,0,-1)
   --i[2];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 0;
      di[2] = 1;
      label(labeled, i, diff_using_adj(i, di));
   }
   ++i[2];

   // (0,0,1)
   ++i[2];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 0;
      di[2] = -1;
      label(labeled, i, diff_using_adj(i, di));
   }
   --i[2];

   //
   // Label in the diagonal directions.
   //

   // (-1,-1,0)
   --i[0];
   --i[1];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = 1;
      di[2] = 0;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (-1,1,0)
   i[1] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = -1;
      di[2] = 0;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (1,-1,0)
   i[0] += 2;
   i[1] -= 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 1;
      di[2] = 0;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (1,1,0)
   i[1] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = -1;
      di[2] = 0;
      label(labeled, i, diff_using_diag(i, di));
   }



   // (-1,0,-1)
   i[0] -= 2;
   --i[1];
   --i[2];
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = 0;
      di[2] = 1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (-1,0,1)
   i[2] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 1;
      di[1] = 0;
      di[2] = -1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (1,0,-1)
   i[0] += 2;
   i[2] -= 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 0;
      di[2] = 1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (1,0,1)
   i[2] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = -1;
      di[1] = 0;
      di[2] = -1;
      label(labeled, i, diff_using_diag(i, di));
   }



   // (0,-1,-1)
   --i[0];
   --i[1];
   i[2] -= 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 1;
      di[2] = 1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (0,-1,1)
   i[2] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = 1;
      di[2] = -1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (0,1,-1)
   i[1] += 2;
   i[2] -= 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = -1;
      di[2] = 1;
      label(labeled, i, diff_using_diag(i, di));
   }

   // (0,1,1)
   i[2] += 2;
   if (is_labeled_or_unlabeled(i)) {
      di[0] = 0;
      di[1] = -1;
      di[2] = -1;
      label(labeled, i, diff_using_diag(i, di));
   }
   //std::cerr << "Done labal_neighbors().\n";
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_using_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   return ads::min(_equation.diff_adj(i, di),
                   diff_adj_diag(i, di),
                   diff_adj_diag_diag(i, di));
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_adj_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   IndexList dj = {{di[0] + di[2], di[1] + di[0], di[2] + di[1]}};
   const Number x1 = _equation.diff_adj_diag(i, di, dj);

   dj[0] = di[0] - di[2];
   dj[1] = di[1] - di[0];
   dj[2] = di[2] - di[1];
   const Number x2 = _equation.diff_adj_diag(i, di, dj);

   dj[0] = di[0] + di[1];
   dj[1] = di[1] + di[2];
   dj[2] = di[2] + di[0];
   const Number x3 = _equation.diff_adj_diag(i, di, dj);

   dj[0] = di[0] - di[1];
   dj[1] = di[1] - di[2];
   dj[2] = di[2] - di[0];
   const Number x4 = _equation.diff_adj_diag(i, di, dj);

   return ads::min(x1, x2, x3, x4);
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_adj_diag_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   IndexList dj = {{di[0] + di[2], di[1] + di[0], di[2] + di[1]}};
   IndexList dk = {{di[0] + di[1], di[1] + di[2], di[2] + di[0]}};
   const Number x1 = _equation.diff_adj_diag_diag(i, di, dj, dk);

   dj[0] = di[0] + di[2];
   dj[1] = di[1] + di[0];
   dj[2] = di[2] + di[1];
   dk[0] = di[0] - di[1];
   dk[1] = di[1] - di[2];
   dk[2] = di[2] - di[0];
   const Number x2 = _equation.diff_adj_diag_diag(i, di, dj, dk);

   dj[0] = di[0] - di[2];
   dj[1] = di[1] - di[0];
   dj[2] = di[2] - di[1];
   dk[0] = di[0] + di[1];
   dk[1] = di[1] + di[2];
   dk[2] = di[2] + di[0];
   const Number x3 = _equation.diff_adj_diag_diag(i, di, dj, dk);

   dj[0] = di[0] - di[2];
   dj[1] = di[1] - di[0];
   dj[2] = di[2] - di[1];
   dk[0] = di[0] - di[1];
   dk[1] = di[1] - di[2];
   dk[2] = di[2] - di[0];
   const Number x4 = _equation.diff_adj_diag_diag(i, di, dj, dk);

   return ads::min(x1, x2, x3, x4);
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_using_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(di));
#endif

   return ads::min(_equation.diff_diag(i, di),
                   diff_diag_adj(i, di),
                   diff_diag_diag(i, di),
                   diff_diag_adj_diag(i, di),
                   diff_diag_diag_diag(i, di));
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_diag_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(di));
#endif

   IndexList a = di;
   for (std::size_t n = 0; n != a.size(); ++n) {
      a[n] = std::abs(a[n]);
   }

   IndexList dj = {{di[0] * a[2], di[1] * a[0], di[2] * a[1]}};
   const Number x1 = _equation.diff_diag_adj(i, di, dj);

   dj[0] = di[0] * a[1];
   dj[1] = di[1] * a[2];
   dj[2] = di[2] * a[0];
   const Number x2 = _equation.diff_diag_adj(i, di, dj);

   return std::min(x1, x2);
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_diag_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(di));
#endif

   if (di[0] == 0) {
      IndexList dj = {{1, 0, di[2]}};
      const Number x1 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = -1;
      dj[1] = 0;
      dj[2] = di[2];
      const Number x2 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = 1;
      dj[1] = di[1];
      dj[2] = 0;
      const Number x3 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = -1;
      dj[1] = di[1];
      dj[2] = 0;
      const Number x4 = _equation.diff_diag_diag(i, di, dj);

      return ads::min(x1, x2, x3, x4);
   }
   else if (di[1] == 0) {
      IndexList dj = {{0, 1, di[2]}};
      const Number x1 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = 0;
      dj[1] = -1;
      dj[2] = di[2];
      const Number x2 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = di[0];
      dj[1] = 1;
      dj[2] = 0;
      const Number x3 = _equation.diff_diag_diag(i, di, dj);

      dj[0] = di[0];
      dj[1] = -1;
      dj[2] = 0;
      const Number x4 = _equation.diff_diag_diag(i, di, dj);

      return ads::min(x1, x2, x3, x4);
   }
   // else
   IndexList dj = {{0, di[1], 1}};
   const Number x1 = _equation.diff_diag_diag(i, di, dj);

   dj[0] = 0;
   dj[1] = di[1];
   dj[2] = -1;
   const Number x2 = _equation.diff_diag_diag(i, di, dj);

   dj[0] = di[0];
   dj[1] = 0;
   dj[2] = 1;
   const Number x3 = _equation.diff_diag_diag(i, di, dj);

   dj[0] = di[0];
   dj[1] = 0;
   dj[2] = -1;
   const Number x4 = _equation.diff_diag_diag(i, di, dj);

   return ads::min(x1, x2, x3, x4);
}


template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_diag_adj_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(di));
#endif

   if (di[0] == 0) {
      IndexList dj = {{0, 0, di[2]}};
      IndexList dk = {{1, 0, di[2]}};
      const Number x1 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      //dj[0] = 0;
      //dj[1] = 0;
      //dj[2] = di[2];
      dk[0] = -1;
      //dk[1] = 0;
      //dk[2] = di[2];
      const Number x2 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      //dj[0] = 0;
      dj[1] = di[1];
      dj[2] = 0;
      dk[0] = 1;
      dk[1] = di[1];
      dk[2] = 0;
      const Number x3 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      //dj[0] = 0;
      //dj[1] = di[1];
      //dj[2] = 0;
      dk[0] = -1;
      //dk[1] = di[1];
      //dk[2] = 0;
      const Number x4 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      return ads::min(x1, x2, x3, x4);
   }
   else if (di[1] == 0) {
      IndexList dj = {{0, 0, di[2]}};
      IndexList dk = {{0, 1, di[2]}};
      const Number x1 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      //dj[0] = 0;
      //dj[1] = 0;
      //dj[2] = di[2];
      //dk[0] = 0;
      dk[1] = -1;
      //dk[2] = di[2];
      const Number x2 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      dj[0] = di[0];
      //dj[1] = 0;
      dj[2] = 0;
      dk[0] = di[0];
      dk[1] = 1;
      dk[2] = 0;
      const Number x3 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      //dj[0] = di[0];
      //dj[1] = 0;
      //dj[2] = 0;
      //dk[0] = di[0];
      dk[1] = -1;
      //dk[2] = 0;
      const Number x4 = _equation.diff_diag_adj_diag(i, di, dj, dk);

      return ads::min(x1, x2, x3, x4);
   }
   // else
   IndexList dj = {{0, di[1], 0}};
   IndexList dk = {{0, di[1], 1}};
   const Number x1 = _equation.diff_diag_adj_diag(i, di, dj, dk);

   //dj[0] = 0;
   //dj[1] = di[1];
   //dj[2] = 0;
   //dk[0] = 0;
   //dk[1] = di[1];
   dk[2] = -1;
   const Number x2 = _equation.diff_diag_adj_diag(i, di, dj, dk);

   dj[0] = di[0];
   dj[1] = 0;
   //dj[2] = 0;
   dk[0] = di[0];
   dk[1] = 0;
   dk[2] = 1;
   const Number x3 = _equation.diff_diag_adj_diag(i, di, dj, dk);

   //dj[0] = di[0];
   //dj[1] = 0;
   //dj[2] = 0;
   //dk[0] = di[0];
   //dk[1] = 0;
   dk[2] = -1;
   const Number x4 = _equation.diff_diag_adj_diag(i, di, dj, dk);

   return ads::min(x1, x2, x3, x4);
}



template<typename T, class Equation>
inline
typename DiffSchemeAdjDiag<3, T, Equation>::Number
DiffSchemeAdjDiag<3, T, Equation>::
diff_diag_diag_diag(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(di));
#endif

   if (di[0] == 0) {
      IndexList dj = {{1, di[1], 0}};
      IndexList dk = {{1, 0, di[2]}};
      const Number x1 = _equation.diff_diag_diag_diag(i, di, dj, dk);

      dj[0] = -1;
      //dj[1] = di[1];
      //dj[2] = 0;
      dk[0] = -1;
      //dk[1] = 0;
      //dk[2] = di[2];
      const Number x2 = _equation.diff_diag_diag_diag(i, di, dj, dk);

      return std::min(x1, x2);
   }
   else if (di[1] == 0) {
      IndexList dj = {{di[0], 1, 0}};
      IndexList dk = {{0, 1, di[2]}};
      const Number x1 = _equation.diff_diag_diag_diag(i, di, dj, dk);

      //dj[0] = di[0];
      dj[1] = -1;
      //dj[2] = 0;
      //dk[0] = 0;
      dk[1] = -1;
      //dk[2] = di[2];
      const Number x2 = _equation.diff_diag_diag_diag(i, di, dj, dk);

      return std::min(x1, x2);
   }
   // else
   IndexList dj = {{di[0], 0, 1}};
   IndexList dk = {{0, di[1], 1}};
   const Number x1 = _equation.diff_diag_diag_diag(i, di, dj, dk);

   //dj[0] = di[0];
   //dj[1] = 0;
   dj[2] = -1;
   //dk[0] = 0;
   //dk[1] = di[1];
   dk[2] = -1;
   const Number x2 = _equation.diff_diag_diag_diag(i, di, dj, dk);

   return std::min(x1, x2);
}

} // namespace hj
}

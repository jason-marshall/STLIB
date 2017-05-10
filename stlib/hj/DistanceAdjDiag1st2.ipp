// -*- C++ -*-

#if !defined(__hj_DistanceAdjDiag1st2_ipp__)
#error This file is an implementation detail of the class DistanceAdjDiag1st.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
typename DistanceAdjDiag1st<2, T>::Number
DistanceAdjDiag1st<2, T>::
diff_adj(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(d));
#endif

   return diff_a1(_solution(i[0] + d[0], i[1] + d[1]));
}


template<typename T>
inline
typename DistanceAdjDiag1st<2, T>::Number
DistanceAdjDiag1st<2, T>::
diff_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(a) && debug::is_diagonal(b));
#endif

   if (_status(i[0] + b[0], i[1] + b[1]) == KNOWN) {
      return diff_a1_d1(_solution(i[0] + a[0], i[1] + a[1]),
                        _solution(i[0] + b[0], i[1] + b[1]));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename DistanceAdjDiag1st<2, T>::Number
DistanceAdjDiag1st<2, T>::
diff_diag(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(d));
#endif

   return diff_d1(_solution(i[0] + d[0], i[1] + d[1]));
}


template<typename T>
inline
typename DistanceAdjDiag1st<2, T>::Number
DistanceAdjDiag1st<2, T>::
diff_diag_adj(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) && debug::is_adjacent(b));
#endif

   if (_status(i[0] + b[0], i[1] + b[1]) == KNOWN) {
      return diff_a1_d1(_solution(i[0] + b[0], i[1] + b[1]),
                        _solution(i[0] + a[0], i[1] + a[1]));
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

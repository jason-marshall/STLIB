// -*- C++ -*-

#if !defined(__hj_DistanceAdjDiag2nd2_ipp__)
#error This file is an implementation detail of the class DistanceAdjDiag2nd.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
typename DistanceAdjDiag2nd<2, T>::Number
DistanceAdjDiag2nd<2, T>::
diff_adj(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(d));
#endif

   int
   i1 = i[0] + d[0],
   i2 = i[0] + 2 * d[0],
   j1 = i[1] + d[1],
   j2 = i[1] + 2 * d[1];

   if (_status(i2, j2) == KNOWN &&
         _solution(i1, j1) > _solution(i2, j2)) {
      return diff_a2(_solution(i1, j1), _solution(i2, j2));
   }
   return diff_a1(_solution(i1, j1));
}


template<typename T>
inline
typename DistanceAdjDiag2nd<2, T>::Number
DistanceAdjDiag2nd<2, T>::
diff_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(a) && debug::is_diagonal(b));
#endif

   int
   ai1 = i[0] + a[0],
   ai2 = i[0] + 2 * a[0],
   aj1 = i[1] + a[1],
   aj2 = i[1] + 2 * a[1],
   di1 = i[0] + b[0],
   di2 = i[0] + 2 * b[0],
   dj1 = i[1] + b[1],
   dj2 = i[1] + 2 * b[1];

   if (_status(di1, dj1) == KNOWN) {
      if (_status(ai2, aj2) == KNOWN &&
            _solution(ai1, aj1) > _solution(ai2, aj2)) {
         if (_status(di2, dj2) == KNOWN &&
               _solution(di1, dj1) > _solution(di2, dj2)) {
            return diff_a2_d2(_solution(ai1, aj1),
                              _solution(ai2, aj2),
                              _solution(di1, dj1),
                              _solution(di2, dj2));
         }
         else {
            return diff_a2_d1(_solution(ai1, aj1),
                              _solution(ai2, aj2),
                              _solution(di1, dj1));
         }
      }
      else {
         if (_status(di2, dj2) == KNOWN &&
               _solution(di1, dj1) > _solution(di2, dj2)) {
            return diff_a1_d2(_solution(ai1, aj1),
                              _solution(di1, dj1),
                              _solution(di2, dj2));
         }
         else {
            return diff_a1_d1(_solution(ai1, aj1),
                              _solution(di1, dj1));
         }
      }
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename DistanceAdjDiag2nd<2, T>::Number
DistanceAdjDiag2nd<2, T>::
diff_diag(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(d));
#endif

   int
   i1 = i[0] + d[0],
   i2 = i[0] + 2 * d[0],
   j1 = i[1] + d[1],
   j2 = i[1] + 2 * d[1];

   if (_status(i2, j2) == KNOWN &&
         _solution(i1, j1) > _solution(i2, j2)) {
      return diff_d2(_solution(i1, j1), _solution(i2, j2));
   }
   return diff_d1(_solution(i1, j1));
}


template<typename T>
inline
typename DistanceAdjDiag2nd<2, T>::Number
DistanceAdjDiag2nd<2, T>::
diff_diag_adj(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) && debug::is_adjacent(b));
#endif

   int
   di1 = i[0] + a[0],
   di2 = i[0] + 2 * a[0],
   dj1 = i[1] + a[1],
   dj2 = i[1] + 2 * a[1],
   ai1 = i[0] + b[0],
   ai2 = i[0] + 2 * b[0],
   aj1 = i[1] + b[1],
   aj2 = i[1] + 2 * b[1];

   if (_status(ai1, aj1) == KNOWN) {
      if (_status(di2, dj2) == KNOWN &&
            _solution(di1, dj1) > _solution(di2, dj2)) {
         if (_status(ai2, aj2) == KNOWN &&
               _solution(ai1, aj1) > _solution(ai2, aj2)) {
            return diff_a2_d2(_solution(ai1, aj1),
                              _solution(ai2, aj2),
                              _solution(di1, dj1),
                              _solution(di2, dj2));
         }
         else {
            return diff_a1_d2(_solution(ai1, aj1),
                              _solution(di1, dj1),
                              _solution(di2, dj2));
         }
      }
      else {
         if (_status(ai2, aj2) == KNOWN &&
               _solution(ai1, aj1) > _solution(ai2, aj2)) {
            return diff_a2_d1(_solution(ai1, aj1),
                              _solution(ai2, aj2),
                              _solution(di1, dj1));
         }
         else {
            return diff_a1_d1(_solution(ai1, aj1),
                              _solution(di1, dj1));
         }
      }
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

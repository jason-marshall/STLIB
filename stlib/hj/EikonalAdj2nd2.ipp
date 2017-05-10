// -*- C++ -*-

#if !defined(__hj_EikonalAdj2nd2_ipp__)
#error This file is an implementation detail of the class EikonalAdj2nd2.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
typename EikonalAdj2nd<2, T>::Number
EikonalAdj2nd<2, T>::
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
      return EquationBase::diff_a2(_solution(i1, j1), _solution(i2, j2),
                                   _inverseSpeed(i));
   }
   return diff_a1(_solution(i1, j1), _inverseSpeed(i));
}


template<typename T>
inline
typename EikonalAdj2nd<2, T>::Number
EikonalAdj2nd<2, T>::
diff_adj_adj(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(a));
   assert(debug::is_adjacent(b));
#endif

   int
   ai1 = i[0] + a[0],
   ai2 = i[0] + 2 * a[0],
   aj1 = i[1] + a[1],
   aj2 = i[1] + 2 * a[1],
   bi1 = i[0] + b[0],
   bi2 = i[0] + 2 * b[0],
   bj1 = i[1] + b[1],
   bj2 = i[1] + 2 * b[1];

   if (_status(bi1, bj1) == KNOWN) {
      if (_status(ai2, aj2) == KNOWN &&
            _solution(ai1, aj1) > _solution(ai2, aj2)) {
         if (_status(bi2, bj2) == KNOWN &&
               _solution(bi1, bj1) > _solution(bi2, bj2)) {
            return EquationBase::diff_a2_a2(_solution(ai1, aj1),
                                            _solution(ai2, aj2),
                                            _solution(bi1, bj1),
                                            _solution(bi2, bj2),
                                            _inverseSpeed(i));
         }
         else {
            return EquationBase::diff_a2_a1(_solution(ai1, aj1),
                                            _solution(ai2, aj2),
                                            _solution(bi1, bj1),
                                            _inverseSpeed(i));
         }
      }
      else {
         if (_status(bi2, bj2) == KNOWN &&
               _solution(bi1, bj1) > _solution(bi2, bj2)) {
            return EquationBase::diff_a2_a1(_solution(bi1, bj1),
                                            _solution(bi2, bj2),
                                            _solution(ai1, aj1),
                                            _inverseSpeed(i));
         }
         else {
            return diff_a1_a1(_solution(bi1, bj1),
                              _solution(ai1, aj1),
                              _inverseSpeed(i));
         }
      }
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

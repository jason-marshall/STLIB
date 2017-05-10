// -*- C++ -*-

#if !defined(__hj_EikonalAdjDiag1st3_ipp__)
#error This file is an implementation detail of the class EikonalAdjDiag1st.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_adj(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(d));
#endif

   return diff_a1(_solution(i + d), _inverseSpeed(i));
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(a) &&
          debug::is_diagonal(b));
#endif

   if (_status(i + b) == KNOWN) {
      return diff_a1_d1(_solution(i + a),
                        _solution(i + b),
                        _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_adj_diag_diag(const IndexList& i, const IndexList& a, const IndexList& b,
                   const IndexList& c) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(a) &&
          debug::is_diagonal(b) &&
          debug::is_diagonal(c));
#endif

   if (_status(i + b) == KNOWN &&
         _status(i + c) == KNOWN) {
      return diff_a1_d1_d1(_solution(i + a),
                           _solution(i + b),
                           _solution(i + c),
                           _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_diag(const IndexList& i, const IndexList& d) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(d));
#endif

   return diff_d1(_solution(i + d), _inverseSpeed(i));
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_diag_adj(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) &&
          debug::is_adjacent(b));
#endif

   if (_status(i + b) == KNOWN) {
      return diff_a1_d1(_solution(i + b),
                        _solution(i + a),
                        _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_diag_diag(const IndexList& i, const IndexList& a, const IndexList& b) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) &&
          debug::is_diagonal(b));
#endif

   if (_status(i + b) == KNOWN) {
      return diff_d1_d1(_solution(i + a),
                        _solution(i + b),
                        _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_diag_adj_diag(const IndexList& i, const IndexList& a, const IndexList& b,
                   const IndexList& c) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) &&
          debug::is_adjacent(b) &&
          debug::is_diagonal(c));
#endif

   if (_status(i + b) == KNOWN &&
         _status(i + c) == KNOWN) {
      return diff_a1_d1_d1(_solution(i + b),
                           _solution(i + a),
                           _solution(i + c),
                           _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename EikonalAdjDiag1st<3, T>::Number
EikonalAdjDiag1st<3, T>::
diff_diag_diag_diag(const IndexList& i, const IndexList& a, const IndexList& b,
                    const IndexList& c) const {
#ifdef STLIB_DEBUG
   assert(debug::is_diagonal(a) &&
          debug::is_diagonal(b) &&
          debug::is_diagonal(c));
#endif

   if (_status(i + b) == KNOWN &&
         _status(i + c) == KNOWN) {
      return diff_d1_d1_d1(_solution(i + a),
                           _solution(i + b),
                           _solution(i + c),
                           _inverseSpeed(i));
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

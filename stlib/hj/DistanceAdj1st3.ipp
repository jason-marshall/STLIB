// -*- C++ -*-

#if !defined(__hj_DistanceAdj1st3_ipp__)
#error This file is an implementation detail of the class DistanceAdj1st.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
typename DistanceAdj1st<3, T>::Number
DistanceAdj1st<3, T>::
diff_adj(const IndexList& i, const IndexList& di) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(di));
#endif

   return diff_a1(_solution(i[0] + di[0], i[1] + di[1], i[2] + di[2]));
}


template<typename T>
inline
typename DistanceAdj1st<3, T>::Number
DistanceAdj1st<3, T>::
diff_adj_adj(const IndexList& i, const IndexList& adi, const IndexList& bdi) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(adi));
   assert(debug::is_adjacent(bdi));
#endif

   if (_status(i[0] + bdi[0], i[1] + bdi[1], i[2] + bdi[2]) == KNOWN) {
      return diff_a1_a1
             (_solution(i[0] + adi[0], i[1] + adi[1], i[2] + adi[2]),
              _solution(i[0] + bdi[0], i[1] + bdi[1], i[2] + bdi[2]));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename DistanceAdj1st<3, T>::Number
DistanceAdj1st<3, T>::
diff_adj_adj_adj(const IndexList& i, const IndexList& adi, const IndexList& bdi,
                 const IndexList& cdi) const {
#ifdef STLIB_DEBUG
   assert(debug::is_adjacent(adi));
   assert(debug::is_adjacent(bdi));
   assert(debug::is_adjacent(cdi));
#endif

   if (_status(i[0] + bdi[0], i[1] + bdi[1], i[2] + bdi[2]) == KNOWN &&
         _status(i[0] + cdi[0], i[1] + cdi[1], i[2] + cdi[2]) == KNOWN) {
      return diff_a1_a1_a1
             (_solution(i[0] + adi[0], i[1] + adi[1], i[2] + adi[2]),
              _solution(i[0] + bdi[0], i[1] + bdi[1], i[2] + bdi[2]),
              _solution(i[0] + cdi[0], i[1] + cdi[1], i[2] + cdi[2]));
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

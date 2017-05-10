// -*- C++ -*-

#if !defined(__hj_EikonalScheme_ipp__)
#error This file is an implementation detail of the class EikonalScheme.
#endif

namespace stlib
{
namespace hj {

//
// File I/O
//

template<typename T>
inline
void
print_inverse_speed_array(std::ostream& out,
                          const container::MultiArrayConstRef<T, 2>& inverseSpeed) {
   typedef typename container::MultiArrayConstRef<T, 2>::IndexList IndexList;
   typedef typename container::MultiArrayConstRef<T, 2>::Index Index;

   const IndexList upper = inverseSpeed.bases() +
      ext::convert_array<Index>(inverseSpeed.extents());
   out << "Inverse speed:" << '\n';
   for (Index j = upper[1] - 1; j >= inverseSpeed.bases()[1]; --j) {
      for (Index i = inverseSpeed.bases()[0]; i < upper[0]; ++i) {
         out << inverseSpeed(i, j) << " ";
      }
      out << '\n';
   }
}


template<typename T>
inline
void
print_inverse_speed_array(std::ostream& out,
                          const container::MultiArrayConstRef<T, 3>& inverseSpeed) {
   typedef typename container::MultiArrayConstRef<T, 3>::IndexList IndexList;
   typedef typename container::MultiArrayConstRef<T, 3>::Index Index;

   const IndexList upper = inverseSpeed.bases() +
      ext::convert_array<Index>(inverseSpeed.extents());
   out << "Inverse speed:" << '\n';
   for (Index k = upper[2] - 1; k >= inverseSpeed.bases()[2]; --k) {
      for (Index j = upper[1] - 1; j >= inverseSpeed.bases()[1]; --j) {
         for (Index i = inverseSpeed.bases()[0]; i < upper[0]; ++i) {
            out << inverseSpeed(i, j, k) << " ";
         }
         out << '\n';
      }
      out << '\n';
   }
}


template<std::size_t N, typename T>
inline
void
EikonalScheme<N, T>::
put(std::ostream& out) const {
   print_inverse_speed_array(out, _inverseSpeed);
}

} // namespace hj
}

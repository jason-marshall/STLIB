// -*- C++ -*-

#if !defined(__hj_Distance3_ipp__)
#error This file is an implementation detail of the class Distance.
#endif

namespace stlib
{
namespace hj {

//
// Finite difference schemes.
//

template<typename T>
inline
typename Distance<3, T>::Number
Distance<3, T>::
diff_d1_d1(Number a, Number b) const {
   if (std::abs(a - b) < _dx_o_sqrt2) {
      return 0.5 *(a + b + std::sqrt(6 * _dx2 - 3 *(a - b)*(a - b)));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename Distance<3, T>::Number
Distance<3, T>::
diff_a1_a1_a1(Number a, Number b, Number c) const {
   Number s = a + b + c;
   Number disc = s * s - 3 * (a * a + b * b + c * c - _dx2);
   if (disc >= 0) {
      Number soln = (s + std::sqrt(disc)) / 3;
      if (soln >= a && soln >= b && soln >= c) {
         return soln;
      }
   }
   return std::numeric_limits<Number>::max();
}

template<typename T>
inline
typename Distance<3, T>::Number
Distance<3, T>::
diff_a1_d1_d1(Number a, Number b, Number c) const {
   // a is adjacent, b and c are diagonal
   if (a > b && a > c) {
      Number disc = _dx2 - (a - b) * (a - b) - (a - c) * (a - c);
      if (disc >= 0) {
         Number soln = a + std::sqrt(disc);
         if (soln > 3 * a - b - c) {
            return soln;
         }
      }
   }
   return std::numeric_limits<Number>::max();
}

template<typename T>
inline
typename Distance<3, T>::Number
Distance<3, T>::
diff_d1_d1_d1(Number a, Number b, Number c) const {
   Number disc = 3 * _dx2 - (a - b) * (a - b) - (b - c) * (b - c) - (c - a) * (c - a);
   if (disc >= 0) {
      Number soln = (a + b + c + 2 * std::sqrt(disc)) / 3;
      if (soln >= 3 * a - b - c &&
            soln >= 3 * b - a - c &&
            soln >= 3 * c - b - a) {
         return soln;
      }
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

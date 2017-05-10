// -*- C++ -*-

#if !defined(__hj_Distance2_ipp__)
#error This file is an implementation detail of the class Distance.
#endif

namespace stlib
{
namespace hj {

template<typename T>
inline
Distance<2, T>::
Distance(const Number dx) :
   _dx(dx),
   _dx2(_dx* _dx),
   _dx_t_sqrt2(_dx* std::sqrt(2.0)),
   _dx_o_sqrt2(_dx / std::sqrt(2.0)) {}

//
// Finite difference schemes.
//

template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a2(const Number a1, const Number a2) const {
   return (4 * a1 - a2 + 2 * _dx) / 3;
}

template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_d2(const Number d1, const Number d2) const {
   return (4 * d1 - d2 + 2 * _dx_t_sqrt2) / 3;
}


template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a1_a1(const Number a, const Number b) const {
   // If the following condition is not satisfied, then the characteristic
   // will come from outside the wedge defined by the two adjacent neighbors.
   // In this case, the value will be higher than one of a and b and thus the
   // difference scheme will not be upwind.  For this case, return infinity.
   if (std::abs(a - b) <= _dx) {
      return 0.5 *(a + b + std::sqrt(2 * _dx2 - (a - b) *(a - b)));
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a1_d1(const Number a, const Number d) const {
   // If the following condition is not satisfied, then the characteristic
   // will come from outside the wedge defined by the adjacent and diagonal
   // neighbors.  (The wedge covers an angle of pi / 4.)
   // In this case, we return infinity.
   Number diff = a - d;
   if (0 <= diff && diff <= _dx_o_sqrt2) {
      return a + std::sqrt(_dx2 - diff * diff);
   }
   return std::numeric_limits<Number>::max();
}



template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a2_a1(const Number a1, const Number a2, const Number b) const {
   Number disc = 13 * _dx2 - (4 * a1 - a2 - 3 * b) *
                 (4 * a1 - a2 - 3 * b);
   if (disc >= 0) {
      Number val = (12 * a1 - 3 * a2 + 4 * b + 2 * std::sqrt(disc)) / 13;
      if (1.5 * val - 2 * a1 + 0.5 * a2 >= 0 && val >= b) {
         return val;
      }
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a1_d2(const Number a, const Number d1, const Number d2) const {
   const Number t = 3 * a - 4 * d1 + d2;
   const Number disc = 5 * _dx2 - t * t;
   if (disc >= 0) {
      const Number s = (2 * a + 4 * d1 - d2 + 2 * std::sqrt(disc)) / 5;
      const Number u_x = s - a;
      const Number u_y = 1.5 * s - 2 * d1 + 0.5 * d2 - u_x;
      if (u_y >= 0 && u_x >= u_y) {
         return s;
      }
   }
   return std::numeric_limits<Number>::max();
}



template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a2_d1(const Number a1, const Number a2, const Number d) const {
   const Number t = 4 * a1 - a2 - 3 * d;
   const Number disc = 10 * _dx2 - t * t;
   if (disc >= 0) {
      const Number s = (8 * a1 - 2 * a2 - d + std::sqrt(disc)) / 5;
      const Number u_x = 1.5 * s - 2 * a1 + 0.5 * a2;
      const Number u_y = s - d - u_x;
      if (u_y >= 0 && u_x >= u_y) {
         return s;
      }
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a2_a2(const Number a1, const Number a2,
           const Number b1, const Number b2) const {
   Number disc = 8 * _dx2
                 - (4 * a1 - a2 - 4 * b1 + b2) * (4 * a1 - a2 - 4 * b1 + b2);
   if (disc >= 0) {
      Number val = (4 * a1 - a2 + 4 * b1 - b2 + std::sqrt(disc)) / 6;
      if (1.5 * val - 2 * a1 + 0.5 * a2 >= 0 &&
            1.5 * val - 2 * b1 + 0.5 * b2 >= 0) {
         return val;
      }
   }
   return std::numeric_limits<Number>::max();
}


template<typename T>
inline
typename Distance<2, T>::Number
Distance<2, T>::
diff_a2_d2(const Number a1, const Number a2,
           const Number d1, const Number d2) const {
   const Number t = 4 * a1 - a2 - 4 * d1 + d2;
   const Number disc = 4 * _dx2 - t * t;
   if (disc >= 0) {
      const Number s = (4 * a1 - a2 + std::sqrt(disc)) / 3;
      const Number u_x = 1.5 * s - 2 * a1 + 0.5 * a2;
      const Number u_y = 1.5 * s - 2 * d1 + 0.5 * d2 - u_x;
      if (u_y >= 0 && u_x >= u_y) {
         return s;
      }
   }
   return std::numeric_limits<Number>::max();
}

} // namespace hj
}

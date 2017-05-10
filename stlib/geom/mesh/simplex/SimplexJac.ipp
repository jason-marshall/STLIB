// -*- C++ -*-

#if !defined(__geom_SimplexJac_ipp__)
#error This file is an implementation detail of the class SimplexJac.
#endif

namespace stlib
{
namespace geom {

namespace internal {

// Consult algebraic_simplex_quality.nb.

// I tried to do this with typenames, but it would not compile.

template<typename T>
inline
void
set(const std::array < std::array<T, 1>, 1 + 1 > & s,
    std::array<T, 1>* determinantGradient) {
   (*determinantGradient)[0] = s[1][0];
}

template<typename T>
inline
void
set(const std::array < std::array<T, 2>, 2 + 1 > & s,
    std::array<T, 2>* determinantGradient) {
   // 2 / sqrt(3)
   const T twoOverSqrt3 = 1.1547005383792517;
   (*determinantGradient)[0] = (s[1][1] - s[2][1]) * twoOverSqrt3;
   (*determinantGradient)[1] = (s[2][0] - s[1][0]) * twoOverSqrt3;
}

template<typename T>
inline
void
set(const std::array < std::array<T, 3>, 3 + 1 > & s,
    std::array<T, 3>* determinantGradient) {
   const T sqrt2 = 1.4142135623730951;

   (*determinantGradient)[0] = sqrt2 *
                               (s[1][1] * (s[3][2] - s[2][2]) +
                                s[2][1] * (s[1][2] - s[3][2]) +
                                s[3][1] * (s[2][2] - s[1][2]));
   (*determinantGradient)[1] = sqrt2 *
                               (s[1][0] * (s[2][2] - s[3][2]) +
                                s[2][0] * (s[3][2] - s[1][2]) +
                                s[3][0] * (s[1][2] - s[2][2]));
   (*determinantGradient)[2] = sqrt2 *
                               (s[1][0] * (s[3][1] - s[2][1]) +
                                s[2][0] * (s[1][1] - s[3][1]) +
                                s[3][0] * (s[2][1] - s[1][1]));
}

}


//
// Manipulators.
//


template<std::size_t N, typename T>
inline
void
SimplexJac<N, T>::
setFunction(const Simplex& s) {
   // The coordinates of the simplex after the first vertex has been
   // translated to the origin.  This Jacobian matrix maps the reference
   // simplex to this simplex.
   for (std::size_t i = 0; i != N; ++i) {
      for (int j = 0; j != N; ++j) {
         _matrix(i, j) = s[j+1][i] - s[0][i];
      }
   }
   // Add the part of the transformation: identity to reference.
   _matrix *= Base::identityToReference();
   // Compute the determinant.
   _determinant = ads::computeDeterminant(_matrix);
}


template<std::size_t N, typename T>
inline
void
SimplexJac<N, T>::
set(const Simplex& s) {
   setFunction(s);
   internal::set(s, &_gradientDeterminant);
}


template<std::size_t N, typename T>
inline
void
SimplexJac<N, T>::
setFunction(const std::array < std::array < Number, N + 1 > , N + 1 > & s) {
   Simplex t;
   projectToLowerDimension(s, &t);
   setFunction(t);
}


template<std::size_t N, typename T>
inline
void
SimplexJac<N, T>::
set(const std::array < std::array < Number, N + 1 > , N + 1 > & s) {
   Simplex t;
   projectToLowerDimension(s, &t);
   set(t);
}


} // namespace geom
}

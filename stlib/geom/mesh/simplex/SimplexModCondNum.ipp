// -*- C++ -*-

#if !defined(__geom_SimplexModCondNum_ipp__)
#error This file is an implementation detail of the class SimplexModCondNum.
#endif

namespace stlib
{
namespace geom {

//
// Mathematical Member Functions
//

// Return the kappa quality metric.
template<std::size_t N, typename T>
inline
T
SimplexModCondNum<N, T>::
operator()(const Number minDeterminant) const {
   return computeFunction(minDeterminant,
                          computeFrobeniusNormSquared(getMatrix()),
                          computeFrobeniusNormSquared(getAdjointMatrix()));
}


// Calculate the gradient of the modified condition number quality metric.
template<std::size_t N, typename T>
inline
void
SimplexModCondNum<N, T>::
computeGradient(const Number minDeterminant, Vertex* gradient) const {
   // Calculate the squared norm of the matrix and its adjoint.
   const Number snj = computeFrobeniusNormSquared(getMatrix());
   const Number sna = computeFrobeniusNormSquared(getAdjointMatrix());

   // Calculate the modified kapppa function.
   const Number k = computeFunction(minDeterminant, snj, sna);

   // Calculate the gradient.
   const Number d = SimplexModDet<T>::getDelta(minDeterminant);
   const Number den =
      std::sqrt(getDeterminant() * getDeterminant() + 4.0 * d * d);
   for (std::size_t n = 0; n != N; ++n) {
      (*gradient)[n] = k *
                       (computeInnerProduct(getGradientMatrix()[n], getMatrix()) / snj +
                        computeInnerProduct(getAdjointGradientMatrix()[n], getAdjointMatrix()) /
                        sna - getGradientDeterminant()[n] / den);
   }
}


template<std::size_t N, typename T>
inline
T
SimplexModCondNum<N, T>::
computeFunction(const Number minDeterminant,
                const Number snj, const Number sna) const {
   // If none of the determinants are small.
   if (minDeterminant >= SimplexModDet<T>::getEpsilon()) {
      // Return the unmodified quality metric.
      return Base::computeFunction(snj, sna);
   }
#ifdef STLIB_DEBUG
   assert(snj >= 0 && sna >= 0);
#endif
   // Else, some of the determinants are small or negative.
   const Number h = getH(minDeterminant);
   // CONTINUE: What should I do when the numerator vanishes?
   const Number numerator = std::sqrt(snj) * std::sqrt(sna);
   if (numerator != 0) {
      return numerator / (N * h);
   }
   return std::numeric_limits<Number>::max();
}

} // namespace geom
}

// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorIfmAcNorm_ipp__)
#error This file is an implementation detail of PoissonGeneratorIfmAcNorm.
#endif

namespace stlib
{
namespace numerical {


//! Threshhold for whether one should use the inversion from the mode method or the acceptance-complement method in computing a Poisson deviate.
/*!
  For the specializations I tested the code on an Intel core duo, compiled
  with GNU g++ 4.0 using the flags: -O3 -funroll-loops -fstrict-aliasing.
*/
template<typename T, class Generator>
class PdianIfmVsAc {
public:
   //! Use the inversion from the mode method for means less than this value.
   static
   T
   getThreshhold() {
      return 45;
   }
};



// Construct using the normal generator and the threshhold.
template<class _Uniform, template<class> class Normal, typename _Result>
PoissonGeneratorIfmAcNorm<_Uniform, Normal, _Result>::
PoissonGeneratorIfmAcNorm(NormalGenerator* normalGenerator,
                          Number normalThreshhold) :
   _inversionFromTheMode(normalGenerator->getDiscreteUniformGenerator(),
                         PdianIfmVsAc<Number, DiscreteUniformGenerator>::
                         getThreshhold()),
   _acceptanceComplementWinrand(normalGenerator),
   _normal(normalGenerator),
   _normalThreshhold(normalThreshhold) {}


template<class _Uniform, template<class> class Normal, typename _Result>
inline
typename PoissonGeneratorIfmAcNorm<_Uniform, Normal, _Result>::result_type
PoissonGeneratorIfmAcNorm<_Uniform, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is small, use the inversion from the mode method.
   if (mean < PdianIfmVsAc<Number, DiscreteUniformGenerator>::getThreshhold()) {
      return _inversionFromTheMode(mean);
   }
   // Use acceptance-complement for medium values.
   if (mean < _normalThreshhold) {
      return _acceptanceComplementWinrand(mean);
   }
   // Use normal approximation for large means.
   return _normal(mean);
}

} // namespace numerical
}

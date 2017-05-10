// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorInvAcNormSure_ipp__)
#error This file is an implementation detail of PoissonGeneratorInvAcNormSure.
#endif

namespace stlib
{
namespace numerical {

// Construct using the normal generator and the threshhold.
template<class _Uniform, template<class> class Normal, typename _Result>
PoissonGeneratorInvAcNormSure<_Uniform, Normal, _Result>::
PoissonGeneratorInvAcNormSure(NormalGenerator* normalGenerator,
                              Number normalThreshhold, Number sureThreshhold) :
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   _inversion(normalGenerator->getDiscreteUniformGenerator(),
              getAcThreshhold()),
#else
   _inversion(normalGenerator->getDiscreteUniformGenerator()),
#endif
   _acceptanceComplementWinrand(normalGenerator),
   _normal(normalGenerator),
   _normalThreshhold(normalThreshhold),
   _sureThreshhold(sureThreshhold) {}

template<class _Uniform, template<class> class Normal, typename _Result>
inline
typename PoissonGeneratorInvAcNormSure<_Uniform, Normal, _Result>::result_type
PoissonGeneratorInvAcNormSure<_Uniform, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is small, use the inversion method.
   if (mean < getAcThreshhold()) {
      return _inversion(mean);
   }
   // Use acceptance-complement for medium values.
   if (mean < _normalThreshhold) {
      return _acceptanceComplementWinrand(mean);
   }
   // Use normal approximation for large means.
   if (mean < _sureThreshhold) {
      return _normal(mean);
   }
   // Use a sure number for very large means. Round to the nearest integer.
   return numerical::roundNonNegative<result_type>(mean);
}

} // namespace numerical
}

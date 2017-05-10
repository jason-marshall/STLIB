// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorInvAcNorm_ipp__)
#error This file is an implementation detail of PoissonGeneratorInvAcNorm.
#endif

namespace stlib
{
namespace numerical {


//! Threshhold for whether one should use the inversion method or the acceptance-complement method in computing a Poisson deviate.
/*!
  For the specializations I tested the code on an Intel core duo, compiled
  with GNU g++ 4.0 using the flags: -O3 -funroll-loops -fstrict-aliasing.
*/
template<typename T, class Generator>
class PdianInvVsAc {
public:
   //! Use the inversion method for means less than this value.
   static
   T
   getThreshhold() {
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      return 13;
#else
      return 6.5;
#endif
   }
};



// Construct using the normal generator and the threshhold.
template<class _Uniform, template<class> class Normal, typename _Result>
PoissonGeneratorInvAcNorm<_Uniform, Normal, _Result>::
PoissonGeneratorInvAcNorm(NormalGenerator* normalGenerator,
                          Number normalThreshhold) :
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   _inversion(normalGenerator->getDiscreteUniformGenerator(),
              PdianInvVsAc<Number, _Uniform>::getThreshhold()),
#else
   _inversion(normalGenerator->getDiscreteUniformGenerator()),
#endif
   _acceptanceComplementWinrand(normalGenerator),
   _normal(normalGenerator),
   _normalThreshhold(normalThreshhold) {}


template<class _Uniform, template<class> class Normal, typename _Result>
inline
typename PoissonGeneratorInvAcNorm<_Uniform, Normal, _Result>::result_type
PoissonGeneratorInvAcNorm<_Uniform, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is small, use the inversion method.
   if (mean < PdianInvVsAc<Number, DiscreteUniformGenerator>::getThreshhold()) {
      return _inversion(mean);
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

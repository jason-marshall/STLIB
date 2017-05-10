// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorInvIfmAcNorm_ipp__)
#error This file is an implementation detail of PoissonGeneratorInvIfmAcNorm.
#endif

namespace stlib
{
namespace numerical {


//! Threshhold for whether one should use the inversion method or the inversion from the mode method in computing a Poisson deviate.
/*!
  For the specializations I tested the code on an Intel core duo, compiled
  with GNU g++ 4.0 using the flags: -O3 -funroll-loops -fstrict-aliasing.
*/
template<typename T, class Generator>
class PdiianInvVsIfm {
public:
   //! Use the inversion method for means less than this value.
   static
   T
   getThreshhold() {
      return 7;
   }
};

//! Threshhold for whether one should use the inversion from the mode method or the acceptance-complement method in computing a Poisson deviate.
/*!
  For the specializations I tested the code on an Intel core duo, compiled
  with GNU g++ 4.0 using the flags: -O3 -funroll-loops -fstrict-aliasing.
*/
template<typename T, class Generator>
class PdiianIfmVsAc {
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
PoissonGeneratorInvIfmAcNorm<_Uniform, Normal, _Result>::
PoissonGeneratorInvIfmAcNorm(NormalGenerator* normalGenerator,
                             Number normalThreshhold) :
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   _inversion(normalGenerator->getDiscreteUniformGenerator(),
              PdiianInvVsIfm<Number, _Uniform>::getThreshhold()),
#else
   _inversion(normalGenerator->getDiscreteUniformGenerator()),
#endif
   _inversionFromTheMode(normalGenerator->getDiscreteUniformGenerator(),
                         PdiianIfmVsAc<Number, DiscreteUniformGenerator>::
                         getThreshhold()),
   _acceptanceComplementWinrand(normalGenerator),
   _normal(normalGenerator),
   _normalThreshhold(normalThreshhold) {}


template<class _Uniform, template<class> class Normal, typename _Result>
inline
typename PoissonGeneratorInvIfmAcNorm<_Uniform, Normal, _Result>::result_type
PoissonGeneratorInvIfmAcNorm<_Uniform, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is very small, use the inversion method.
   if (mean <
         PdiianInvVsIfm<Number, DiscreteUniformGenerator>::getThreshhold()) {
      return _inversion(mean);
   }
   // If the mean is small, use the inversion from the mode method.
   if (mean < PdiianIfmVsAc<Number, DiscreteUniformGenerator>::getThreshhold()) {
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

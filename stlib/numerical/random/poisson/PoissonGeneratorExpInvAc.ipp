// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorExpInvAc_ipp__)
#error This file is an implementation detail of PoissonGeneratorExpInvAc.
#endif

namespace stlib
{
namespace numerical {


//! Threshhold for whether one should use the exponential inter-arrival method or the inversion method in computing a Poisson deviate.
template<typename T, class Generator>
class PdeiaExpVsInv {
public:
   //! Use the exponential inter-arrival method for means less than this value.
   static
   T
   getThreshhold() {
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      return 0.4;
#else
      return 2.0;
#endif
   }
};


//! Threshhold for whether one should use the inversion method or the acceptance-complement method in computing a Poisson deviate.
template<typename T, class Generator>
class PdeiaInvVsAc {
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


// Construct using the exponential generator and the normal generator.
template < class _Uniform, template<class> class _Exponential,
         template<class> class Normal, typename _Result >
inline
PoissonGeneratorExpInvAc<_Uniform, _Exponential, Normal, _Result>::
PoissonGeneratorExpInvAc(ExponentialGenerator* exponentialGenerator,
                         NormalGenerator* normalGenerator) :
   _exponentialInterArrival(exponentialGenerator),
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   _inversion(exponentialGenerator->getDiscreteUniformGenerator(),
              PdeiaInvVsAc<Number, DiscreteUniformGenerator>::getThreshhold()),
#else
   _inversion(exponentialGenerator->getDiscreteUniformGenerator()),
#endif
   _acceptanceComplementWinrand(normalGenerator) {}



template < class _Uniform, template<class> class _Exponential,
         template<class> class Normal, typename _Result >
inline
typename PoissonGeneratorExpInvAc<_Uniform, _Exponential, Normal, _Result>::result_type
PoissonGeneratorExpInvAc<_Uniform, _Exponential, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is very small, use the exponential inter-arrival method.
   if (mean < PdeiaExpVsInv<Number, DiscreteUniformGenerator>::getThreshhold()) {
      return _exponentialInterArrival(mean);
   }
   // Use the inversion method for small means.
   if (mean < PdeiaInvVsAc<Number, DiscreteUniformGenerator>::getThreshhold()) {
      return _inversion(mean);
   }
   // Use acceptance-complement for the rest.
   return _acceptanceComplementWinrand(mean);
}

} // namespace numerical
}

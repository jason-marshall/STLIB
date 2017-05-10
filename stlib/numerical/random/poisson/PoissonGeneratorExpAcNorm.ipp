// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorExpAcNorm_ipp__)
#error This file is an implementation detail of PoissonGeneratorExpAcNorm.
#endif

namespace stlib
{
namespace numerical {

//! Threshhold for whether one should use the exponential inter-arrival method or the acceptance-complement method in computing a Poisson deviate.
/*!
  For the specializations I tested the code on an Intel core duo, compiled
  with GNU g++ 4.0 using the flags: -O3 -funroll-loops -fstrict-aliasing.
*/
template<class _Generator>
class PoissonExpVsAc {
public:
   //! Use the exponential inter-arrival method for means less than this value.
   enum {Threshhold = 3};
};

template < class _Uniform,
         template<class> class _Exponential,
         template<class> class Normal,
         typename _Result >
inline
typename PoissonGeneratorExpAcNorm<_Uniform, _Exponential, Normal, _Result>::result_type
PoissonGeneratorExpAcNorm<_Uniform, _Exponential, Normal, _Result>::
operator()(const argument_type mean) {
   // If the mean is small, use the exponential inter-arrival method.
   if (mean < PoissonExpVsAc<DiscreteUniformGenerator>::Threshhold) {
      return _exponentialInterArrival(mean);
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

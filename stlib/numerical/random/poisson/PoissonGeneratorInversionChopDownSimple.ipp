// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorInversionChopDownSimple_ipp__)
#error This file is an implementation detail of PoissonGeneratorInversionChopDownSimple.
#endif

namespace stlib
{
namespace numerical {

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorInversionChopDownSimple<_Uniform, _Result>::result_type
PoissonGeneratorInversionChopDownSimple<_Uniform, _Result>::
operator()(const argument_type mean) {
#ifdef STLIB_DEBUG
   // If the mean is too large, we will get underflow in computing p.
   // The algorithm will give incorrect results.
   assert(mean < Number(PoissonGeneratorInversionMaximumMean<Number>::Value));
#endif

   // CONTINUE
   const result_type NumericalFailureBound =
      2 * PoissonGeneratorInversionMaximumMean<Number>::Value;
   const Number exponential = std::exp(-mean);
   while (true) {
      // Poisson random deviate.
      result_type deviate = 0;
      // Scaled probability density function.
      Number pdf = exponential;
      // Uniform deviate.
      Number u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
                 ((*_discreteUniformGenerator)());
      do {
         u -= pdf;
         if (u <= 0) {
            return deviate;
         }
         ++deviate;
         pdf *= mean / deviate;
      }
      while (deviate != NumericalFailureBound);
   }
}

} // namespace numerical
}

// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorExponentialInterArrivalUsingUniform_ipp__)
#error This file is an implementation detail of PoissonGeneratorExponentialInterArrivalUsingUniform.
#endif

namespace stlib
{
namespace numerical {

#ifdef NUMERICAL_POISSON_SMALL_MEAN

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorExponentialInterArrivalUsingUniform<_Uniform, _Result>::result_type
PoissonGeneratorExponentialInterArrivalUsingUniform<_Uniform, _Result>::
operator()(const argument_type mean) {
   // This is a good idea if the mean is zero a significant fraction of the
   // time.  This lets us skip computing an exponential and random number.
   if (mean == 0) {
      return 0;
   }

   // Calculate a uniform random deviate.
   Number t = transformDiscreteDeviateToContinuousDeviateClosed<Number>
              ((*_discreteUniformGenerator)());

   // Recall that exp(-x) = 1 - x + x^2 / 2! - x^3 / 3! + ...
   // Note that 1 - x <= exp(-x) for all x.  The two functions and their first
   // derivatives are equal at x = 0, and the latter is convex (f''(x) > 0).
   // Thus t <= 1 - mean implies that t <= exp(-mean).  We check this condition
   // to avoid having to compute the exponential function for small means.
   if (t <= 1 - mean) {
      return 0;
   }

#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
   // If the mean is new, compute the exponential.
   if (mean != _oldm) {
      _oldm = mean;
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      _g = _expNeg(mean);
#else
      _g = std::exp(- mean);
#endif
   }
#else // NUMERICAL_POISSON_CACHE_OLD_MEAN
   // Compute the exponential.  The leading underscore is for compatability
   // with the above case.
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   const Number _g = _expNeg(mean);
#else
   const Number _g = std::exp(- mean);
#endif
#endif // NUMERICAL_POISSON_CACHE_OLD_MEAN

   int em = 0;
   while (t > _g) {
      ++em;
      t *= transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_discreteUniformGenerator)());
   }

   return em;
}

#else // NUMERICAL_POISSON_SMALL_MEAN

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorExponentialInterArrivalUsingUniform<_Uniform, _Result>::result_type
PoissonGeneratorExponentialInterArrivalUsingUniform<_Uniform, _Result>::
operator()(const argument_type mean) {
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
   // If the mean is new, compute the exponential.
   if (mean != _oldm) {
      _oldm = mean;
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      _g = _expNeg(mean);
#else
_g = std::exp(- mean);
#endif
   }
#else // NUMERICAL_POISSON_CACHE_OLD_MEAN
// Compute the exponential.  The leading underscore is for compatability
// with the above case.
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
const Number _g = _expNeg(mean);
#else
const Number _g = std::exp(- mean);
#endif
#endif // NUMERICAL_POISSON_CACHE_OLD_MEAN

   int em = -1;
   Number t = 1;
   do {
      ++em;
      t *= transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_discreteUniformGenerator)());
   }
   while (t > _g);

   return em;
}

#endif // NUMERICAL_POISSON_SMALL_MEAN

} // namespace numerical
}

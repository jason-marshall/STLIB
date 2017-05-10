// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorRejectionNr_ipp__)
#error This file is an implementation detail of PoissonGeneratorRejectionNr.
#endif

namespace stlib
{
namespace numerical {

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorRejectionNr<_Uniform, _Result>::result_type
PoissonGeneratorRejectionNr<_Uniform, _Result>::
operator()(const argument_type mean) {
   Number em, t, y;

   // If the mean is new then compute some functions that are used below.
   if (mean != _oldm) {
      _oldm = mean;
      _sq = std::sqrt(2.0 * mean);
      _alxm = std::log(mean);
      _g = mean * _alxm - _logarithmOfGamma(mean + 1.0);
   }

   do {
      do {
         y = std::tan(numerical::Constants<Number>::Pi() *
                      transformDiscreteDeviateToContinuousDeviateOpen<Number>
                      ((*_discreteUniformGenerator)()));
         em = _sq * y + mean;
      }
      while (em < 0.0);
      // The floor function is costly.
      //em = std::floor(em);
      em = numerical::floorNonNegative<result_type>(em);
      t = 0.9 * (1.0 + y * y) * std::exp(em * _alxm -
                                         _logarithmOfGamma(em + 1.0) - _g);
   }
   while (transformDiscreteDeviateToContinuousDeviateClosed<Number>
          ((*_discreteUniformGenerator)()) > t);

   return numerical::floorNonNegative<result_type>(em);
}

} // namespace numerical
}

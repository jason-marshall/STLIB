// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorDirectNr_ipp__)
#error This file is an implementation detail of PoissonGeneratorDirectNr.
#endif

namespace stlib
{
namespace numerical {

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorDirectNr<_Uniform, _Result>::result_type
PoissonGeneratorDirectNr<_Uniform, _Result>::
operator()(const argument_type mean) {
   // If the mean is new, compute the exponential.
   if (mean != _oldm) {
      _oldm = mean;
      _g = std::exp(- mean);
   }

   result_type em = -1;
   Number t = 1.0;
   do {
      ++em;
      t *= transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_discreteUniformGenerator)());
   }
   while (t > _g);

   return em;
}

} // namespace numerical
}

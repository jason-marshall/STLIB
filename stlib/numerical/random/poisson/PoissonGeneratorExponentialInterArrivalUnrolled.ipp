// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorExponentialInterArrivalUnrolled_ipp__)
#error This file is an implementation detail of PoissonGeneratorExponentialInterArrivalUnrolled.
#endif

namespace stlib
{
namespace numerical {


template<class _Uniform, template<class> class _Exponential, typename _Result>
inline
typename PoissonGeneratorExponentialInterArrivalUnrolled<_Uniform, _Exponential, _Result>::result_type
PoissonGeneratorExponentialInterArrivalUnrolled<_Uniform, _Exponential, _Result>::
operator()(argument_type mean) {
   mean -= (*_exponentialGenerator)();
   if (mean < 0) {
      return 0;
   }
   mean -= (*_exponentialGenerator)();
   if (mean < 0) {
      return 1;
   }

   return 2;
}


} // namespace numerical
}

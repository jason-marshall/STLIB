// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorDirectRejectionNr_ipp__)
#error This file is an implementation detail of PoissonGeneratorDirectRejectionNr.
#endif

namespace stlib
{
namespace numerical {

template<class _Uniform, typename _Result>
inline
typename PoissonGeneratorDirectRejectionNr<_Uniform, _Result>::result_type
PoissonGeneratorDirectRejectionNr<_Uniform, _Result>::
operator()(const argument_type mean) {
   // If the mean is small, use the direct method.
   if (mean < 12.0) {
      return _directNr(mean);
   }
   // Otherwise, use the rejection method.
   return _rejectionNr(mean);
}

} // namespace numerical
}

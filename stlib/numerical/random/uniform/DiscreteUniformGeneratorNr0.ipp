// -*- C++ -*-

#if !defined(__numerical_random_DiscreteUniformGeneratorNr0_ipp__)
#error This file is an implementation detail of DiscreteUniformGeneratorNr0.
#endif

namespace stlib
{
namespace numerical {

inline
DiscreteUniformGeneratorNr0::result_type
DiscreteUniformGeneratorNr0::
operator()() {
   const int
   IA = 16807,
   IM = 2147483647,
   IQ = 127773,
   IR = 2836;

#ifdef STLIB_DEBUG
   // If _idum is ever zero, it will always be zero.
   assert(_idum != 0);
#endif
   int k = _idum / IQ;
   _idum = IA * (_idum - k * IQ) - IR * k;
   if (_idum < 0) {
      _idum += IM;
   }
   return _idum;
}


} // namespace numerical
}

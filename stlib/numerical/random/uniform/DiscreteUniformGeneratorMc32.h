// -*- C++ -*-

/*!
  \file numerical/random/uniform/DiscreteUniformGeneratorMc32.h
  \brief Uniform random deviates using the multiplicative congruential method.
*/

#if !defined(__numerical_DiscreteUniformGeneratorMc32_h__)
#define __numerical_DiscreteUniformGeneratorMc32_h__

#include <boost/config.hpp>

#include <limits>
#include <cassert>

namespace stlib
{
namespace numerical {

//! The super-duper generator.
/*!
  For odd seeds, the period of the generator is 2^30.  (CONTINUE Check this.)
*/
struct TraitsMcMarsaglia {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 69069U;
};

//! Fishman and Moore generator 1.
struct TraitsMcFishmanMoore1 {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 1099087573U;
};

//! Fishman and Moore generator 2.
struct TraitsMcFishmanMoore2 {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 2396548189U;
};

//! Fishman and Moore generator 3.
struct TraitsMcFishmanMoore3 {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 2824527309U;
};

//! Fishman and Moore generator 4.
struct TraitsMcFishmanMoore4 {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 3934873077U;
};

//! Fishman and Moore generator 5.
struct TraitsMcFishmanMoore5 {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 392314069U;
};

//! Ahrens and Deiter generator.
struct TraitsMcAhrensDeiter {
   //! Multiplicative parameter.
   BOOST_STATIC_CONSTEXPR unsigned A = 392314069U;
};

//! Uniform random deviates using the multiplicative congruential method.
/*!
  For documentation go to the
  \ref numerical_random_uniform "uniform deviates page".
*/
template < typename Traits = TraitsMcMarsaglia >
class DiscreteUniformGeneratorMc32 {
private:

   unsigned _deviate;

public:

   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef unsigned result_type;


   //! Construct and seed.
   /*!
     If no seed is specified, it is seeded with 1.
   */
   explicit
   DiscreteUniformGeneratorMc32(const unsigned seed = 1) :
      _deviate(seed) {
   }

   //! Copy constructor.
   DiscreteUniformGeneratorMc32(const DiscreteUniformGeneratorMc32& other) :
      _deviate(other._deviate) {
   }

   //! Assignment operator.
   DiscreteUniformGeneratorMc32&
   operator=(const DiscreteUniformGeneratorMc32& other) {
      if (this != &other) {
         _deviate = other._deviate;
      }
      return *this;
   }

   //! Destructor.
   ~DiscreteUniformGeneratorMc32() {}

   //! Seed this random number generator.
   void
   seed(const unsigned s) {
      _deviate = s;
      if (_deviate == 0) {
         _deviate = 1;
      }
   }

   //! Return a discrete uniform random deviate.
   result_type
   operator()() {
#ifdef STLIB_DEBUG
      assert(_deviate != 0);
#endif
      return _deviate *= Traits::A;
   }
};


} // namespace numerical
}

#endif

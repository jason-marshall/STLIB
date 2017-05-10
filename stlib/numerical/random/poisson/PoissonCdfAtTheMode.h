// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonCdfAtTheMode.h
  \brief Cumulative distribution function evaluated at the mode for the Poisson distribution.
*/

#if !defined(__numerical_PoissonCdfAtTheMode_h__)
#define __numerical_PoissonCdfAtTheMode_h__

#include "stlib/numerical/random/poisson/PoissonCdf.h"

#include "stlib/numerical/interpolation/hermite.h"

#include <vector>

namespace stlib
{
namespace numerical {

//! Cumulative distribution function evaluated at the mode for the Poisson distribution.
/*!
  \param T The number type.  By default it is double.

  This functor computes
  \f[
  f(x) = \mathrm{cdf}(x, \lfloor x \rfloor)
  \f]
  where cdf() is the Poisson cumulative distribution function
  \f[
  \mathrm{cdf}(\mu, n) = \sum_{m=0}^n \frac{e^{-\mu} \mu^m}{m!}.
  \f]
*/
template < typename T = double >
class PoissonCdfAtTheMode :
   public std::unary_function<T, T> {
   //
   // Private types.
   //
private:

   typedef std::unary_function<T, T> Base;

   //
   // Public types.
   //
public:

   //! The argument type.
   typedef typename Base::argument_type argument_type;
   //! The result type.
   typedef typename Base::result_type result_type;
   //! The number type.
   typedef T Number;

   //
   // Member data.
   //
private:

   std::vector<Number> _function0, _function1, _derivative0, _derivative1;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   PoissonCdfAtTheMode();

public:

   //! Size constructor.
   PoissonCdfAtTheMode(std::size_t tableSize);

   //! Copy constructor.
   PoissonCdfAtTheMode(const PoissonCdfAtTheMode& other) :
      _function0(other._function0),
      _function1(other._function1),
      _derivative0(other._derivative0),
      _derivative1(other._derivative1) {}

   //! Assignment operator.
   PoissonCdfAtTheMode&
   operator=(const PoissonCdfAtTheMode& other) {
      if (this != &other) {
         _function0 = other._function0;
         _function1 = other._function1;
         _derivative0 = other._derivative0;
         _derivative1 = other._derivative1;
      }
      return *this;
   }

   //! Trivial destructor.
   ~PoissonCdfAtTheMode() {}

   //! Return the cumulative distribution function evaluated at the mode.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonCdfAtTheMode_ipp__
#include "stlib/numerical/random/poisson/PoissonCdfAtTheMode.ipp"
#undef __numerical_random_PoissonCdfAtTheMode_ipp__

#endif

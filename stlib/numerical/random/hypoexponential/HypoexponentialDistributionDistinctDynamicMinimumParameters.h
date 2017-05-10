// -*- C++ -*-

/*!
  \file numerical/random/exponential/HypoexponentialDistributionDistinctDynamicMinimumParameters.h
  \brief Hypoexponential distribution.
*/

#if !defined(__numerical_HypoexponentialDistributionDistinctDynamicMinimumParameters_h__)
#define __numerical_HypoexponentialDistributionDistinctDynamicMinimumParameters_h__

#include <algorithm>
#include <limits>
#include <vector>

#include <cmath>
#include <cassert>

namespace stlib
{
namespace numerical {

//! Hypoexponential distribution.
/*!
  See the
  \ref numerical_random_hypoexponential "hypoexponential distribution page"
  for general information.
*/
class HypoexponentialDistributionDistinctDynamicMinimumParameters {
private:

   //
   // Member data.
   //

   //! The maximum number of allowed parameters.
   std::size_t _capacity;
   //! The minimum parameter value.
   double _minParameter;
   //! The maximum parameter value.
   double _maxParameter;
   //! log(Gamma(n)) where n is the number of parameters.
   double _logGamma;
   //! The parameters.
   std::vector<double> _parameters;
   //! The coefficients used to calculate the probability distribution.
   std::vector<double> _coefficients;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   HypoexponentialDistributionDistinctDynamicMinimumParameters();

public:

   //! Construct from the maximum number of parameters.
   HypoexponentialDistributionDistinctDynamicMinimumParameters
   (const std::size_t capacity) :
      _capacity(capacity),
      _minParameter(0),
      _maxParameter(std::numeric_limits<double>::max()),
      _logGamma(0),
      _parameters(),
      _coefficients() {
      assert(capacity > 0);
   }

   // The default copy constructor and assignment operator are fine.

   //! Destructor.
   ~HypoexponentialDistributionDistinctDynamicMinimumParameters() {}

   //! Clear the set of parameters.
   void
   clear() {
      _minParameter = 0;
      _maxParameter = std::numeric_limits<double>::max();
      _logGamma = 0;
      _parameters.clear();
      _coefficients.clear();
   }

   //! Return the number of parameters.
   std::size_t
   size() const {
      return _parameters.size();
   }

   //! Return the complementary CDF.
   double
   ccdf(const double t) const {
      if (t <= 0 || _parameters.empty()) {
         return 1;
      }
      double result = 0;
      for (std::size_t i = 0; i != _parameters.size(); ++i) {
         result += _coefficients[i] * std::exp(- _parameters[i] * t);
      }
      return result;
   }

   //! Return true if the complementary CDF may be greater than the machine epsilon.
   /*!
     We can bound the CCDF of the hypoexponential distribution with the CCDF
     of the Erlang distribution. Let \f$\lambda\f$ be the minimum value of the
     \e n parameters in the hypoexponential distribution. We use the Erlang
     distribution with shape \e n and rate parameter \f$\lambda\f$.
     \f[
     \mathrm{CCDF}(t) = 1 - \frac{\gamma(n, \lambda t)}{(n - 1)!}
     \f]
     \f[
     \mathrm{CCDF}(t) = \frac{\Gamma(n) - \gamma(n, \lambda t)}{(n - 1)!}
     \f]
     \f[
     \mathrm{CCDF}(t) = \frac{\Gamma(n, \lambda t)}{(n - 1)!}
     \f]
     \f[
     \mathrm{CCDF}(t) \sim
     \frac{(\lambda t)^{n-1} \mathrm{e}^{- \lambda t}}{(n - 1)!}, \quad
     \mathrm{as } \lambda t \rightarrow \infty
     \f]
     \f[
     \log(\mathrm{CCDF}(t)) \sim
     (n - 1) \log(\lambda t) - \lambda t - \log((n - 1)!)
     \quad \mathrm{as } \lambda t \rightarrow \infty
     \f]
   */
   bool
   isCcdfNonzero(const double t) const {
      const double logEpsilon = std::log(std::numeric_limits<double>::epsilon());
      // Special case of no parameters.
      if (_parameters.empty()) {
         return false;
      }
      // Special case of small argument where the asymptotic approximation is
      // not valid.
      if (_minParameter * t < 10) {
         return true;
      }
      // General case.
      return (_parameters.size() - 1) * std::log(_minParameter * t)
             - _minParameter * t - _logGamma > logEpsilon;
   }

   //! Try inserting a parameter.
   /*!
     If the new parameter is distinct and the capacity has not been reached,
     then insert the parameter and return std::numeric_limits<double>::max().
     Otherwise return the argument.
   */
   double
   insert(double p) {
      if (_parameters.size() < _capacity && isDistinct(p)) {
         // Insert the parameter.
         insertParameter(p);
         // Indicate that the parameter was inserted.
         p = std::numeric_limits<double>::max();
         // Recalculate the maximum parameter value and the coefficients.
         rebuild();
      }
      return p;
   }

   //! Try inserting or replacing a parameter.
   /*!
     If the new parameter is distinct and smaller that the maximum
     parameter, then insert the parameter. In this case either return
     the parameter that was removed or return
     std::numeric_limits<double>::max() if one was not removed.

     Otherwise return the argument.
   */
   double
   insertOrReplace(double p) {
      if (p < _maxParameter && isDistinct(p)) {
         if (_parameters.size() < _capacity) {
            // Insert the parameter.
            insertParameter(p);
            // Indicate that no parameter was removed.
            p = std::numeric_limits<double>::max();
         }
         else {
            // Determine which parameter to replace.
            std::size_t index =
               std::max_element(_parameters.begin(), _parameters.end()) -
               _parameters.begin();
            // Insert the new parameter and remove the maximum one.
            std::swap(p, _parameters[index]);
         }
         // Recalculate the maximum parameter value and the coefficients.
         rebuild();
      }
      return p;
   }

private:

   //! Insert the parameter and update _logGamma.
   void
   insertParameter(const double p) {
      // Insert the parameter.
      _parameters.push_back(p);
      _coefficients.push_back(0);
      // Compute the logarithm of the Gamma function of the size
      _logGamma = 1;
      for (std::size_t i = 1; i != _parameters.size(); ++i) {
         _logGamma *= i;
      }
      _logGamma = std::log(_logGamma);
   }

   //! Return true if the argument is distinct from each parameter.
   bool
   isDistinct(const double p) const {
      // The epsilon used to determine if parameters are distinct is the square
      // root of the machine epsilon.
      const double Epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
      for (std::size_t i = 0; i != _parameters.size(); ++i) {
         if (std::abs(p - _parameters[i]) < (p + _parameters[i]) * Epsilon) {
            return false;
         }
      }
      return true;
   }

   //! Recalculate the maximum parameter value and the coefficients.
   void
   rebuild() {
#ifdef STLIB_DEBUG
      assert(! _parameters.empty() && _parameters.size() == _coefficients.size());
#endif
      _minParameter = *std::min_element(_parameters.begin(), _parameters.end());
      if (_parameters.size() == _capacity) {
         _maxParameter = *std::max_element(_parameters.begin(), _parameters.end());
      }
      else {
         // Indicate that there are empty slots for parameters.
         _maxParameter = std::numeric_limits<double>::max();
      }
      for (std::size_t i = 0; i != _coefficients.size(); ++i) {
         _coefficients[i] = 1;
         // Collect the denominator to avoid unecessary divisions.
         double denominator = 1;
         for (std::size_t j = 0; j != _coefficients.size(); ++j) {
            if (j == i) {
               continue;
            }
            _coefficients[i] *= _parameters[j];
            denominator *= _parameters[j] - _parameters[i];
         }
         _coefficients[i] /= denominator;
      }
   }
};

} // namespace numerical
}

#endif

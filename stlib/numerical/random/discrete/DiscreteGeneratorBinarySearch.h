// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinarySearch.h
  \brief Discrete deviate.  Binary search.
*/

#if !defined(__numerical_DiscreteGeneratorBinarySearch_h__)
#define __numerical_DiscreteGeneratorBinarySearch_h__

#include "stlib/numerical/random/discrete/DgPmf.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <boost/config.hpp>

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Binary search.
/*!
  CONTINUE.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinarySearch :
   public DgPmf<false> {
   //
   // Private types.
   //
private:

   typedef DgPmf<false> Base;

   //
   // Public constants.
   //
public:

   //! The sum of the PMF is not automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = false;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef Generator DiscreteUniformGenerator;
   //! The continuous uniform generator.
   typedef ContinuousUniformGeneratorClosed<DiscreteUniformGenerator>
   ContinuousUniformGenerator;
   //! The number type.
   typedef typename Base::Number Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;

   //
   // Member data.
   //
private:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! Cumulative distribution function.  (This is scaled and may not approach unity.)
   std::vector<Number> _cdf;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinarySearch();


   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinarySearch(DiscreteUniformGenerator* generator) :
      Base(),
      _continuousUniformGenerator(generator),
      _cdf() {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinarySearch(DiscreteUniformGenerator* generator,
                                 ForwardIterator begin,
                                 ForwardIterator end) :
      Base(),
      _continuousUniformGenerator(generator),
      _cdf() {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearch
   (const DiscreteGeneratorBinarySearch& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _cdf(other._cdf) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearch&
   operator=(const DiscreteGeneratorBinarySearch& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _cdf = other._cdf;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinarySearch() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _continuousUniformGenerator.seed(seedValue);
   }

   //! Return a discrete deviate.
   result_type
   operator()() {
      // Note that this search will not find an event that has zero probability.
      // There is no need to loop until we get a valid deviate.
      return std::lower_bound(_cdf.begin(), _cdf.end(),
                              _continuousUniformGenerator() * _cdf.back()) -
             _cdf.begin();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability with the specified index.
   using Base::operator[];
   //! Get the number of possible deviates.
   using Base::size;

   //! Get the sum of the probability mass functions.
   /*!
     \pre The CDF must be computed before calling this function.
   */
   Number
   sum() const {
      return _cdf.back();
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return sum() > 0;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the probability mass function with the specified index.
   using Base::set;

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      Base::initialize(begin, end);
      _cdf.resize(size());
      updateSum();
   }

   //! Recompute the CDF.
   /*!
     This must be called after modifying the PMF and before drawing a deviate.
   */
   void
   updateSum() {
      // Compute the cumulative distribution function.
      std::partial_sum(Base::begin(), Base::end(), _cdf.begin());
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      Base::print(out);
      out << "CDF =\n" << _cdf << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif

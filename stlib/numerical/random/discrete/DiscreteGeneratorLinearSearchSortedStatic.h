// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearchSortedStatic.h
  \brief discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearchSortedStatic_h__)
#define __numerical_DiscreteGeneratorLinearSearchSortedStatic_h__

#include "stlib/numerical/random/discrete/DgPmfAndSumOrderedPairPointer.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ads/counter/CounterWithReset.h"

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Linear search.
/*!
  \param Generator is the discrete, uniform generator.

  This class determines the deviate with a linear search on the sorted
  probabilities.

  \note This is not a base class for the dynamic version.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorLinearSearchSortedStatic :
   public DgPmfOrderedPairPointer<true> {
   //
   // Private types.
   //
private:

   typedef DgPmfOrderedPairPointer<true> Base;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef _Generator DiscreteUniformGenerator;
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
protected:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The sum of the PMF.
   Number _sum;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearchSortedStatic();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearchSortedStatic(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _sum(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearchSortedStatic(DiscreteUniformGenerator* generator,
         ForwardIterator begin,
         ForwardIterator end) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _sum(0) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchSortedStatic
   (const DiscreteGeneratorLinearSearchSortedStatic& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _sum(other._sum) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchSortedStatic&
   operator=(const DiscreteGeneratorLinearSearchSortedStatic& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _sum = other._sum;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearchSortedStatic() {}

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
      // Loop until we get a valid deviate (non-zero weighted probability).
      result_type index;
      do {
         index = linearSearchChopDownGuardedPair
                 (Base::begin(), Base::end(), _continuousUniformGenerator() * sum());
      }
      while (operator[](index) == 0);
      return index;
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
   Number
   sum() const {
      return _sum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return _sum > 0;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DiscreteGeneratorLinearSearchSortedStatic& other) {
      return Base::operator==(other) &&
             _continuousUniformGenerator == other._continuousUniformGenerator &&
             _sum == other._sum;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator start, ForwardIterator finish) {
      // Initialize the PMF.
      Base::initialize(start, finish);
      // Compute the PMF sum.
      _sum = 0;
      const Base::const_iterator end = Base::end();
      for (Base::const_iterator i = Base::begin(); i != end; ++i) {
         _sum += i->first;
      }
      // Sort the PMF.
      rebuild();
   }

private:

   //! Sort the PMF.
   void
   rebuild() {
      std::sort(Base::begin(), Base::end(), Base::ValueGreater());
      Base::computePointers();
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
      out << "PMF sum = " << _sum << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif

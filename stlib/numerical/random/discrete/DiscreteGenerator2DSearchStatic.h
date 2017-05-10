// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGenerator2DSearchStatic.h
  \brief discrete deviate.  CDF inversion using a partial sums of the PMF.
*/

#if !defined(__numerical_DiscreteGenerator2DSearchStatic_h__)
#define __numerical_DiscreteGenerator2DSearchStatic_h__

#include "stlib/numerical/random/discrete/DgPmf.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  CDF inversion using 2-D search.
/*!
  \param Generator is the discrete, uniform generator.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGenerator2DSearchStatic :
   public DgPmf<true> {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function.
   typedef DgPmf<true> Base;

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
protected:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The sum of the PMF.
   Number _sum;
   //! Partial sums of the PMF.
   std::vector<Number> _partialPmfSums;
   //! The elements per partial sum.
   std::size_t _elementsPerPartialSum;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGenerator2DSearchStatic();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGenerator2DSearchStatic(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _sum(0),
      // Empty array.
      _partialPmfSums(),
      _elementsPerPartialSum(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGenerator2DSearchStatic(DiscreteUniformGenerator* generator,
                                   ForwardIterator begin, ForwardIterator end) :
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _sum(0),
      _partialPmfSums(),
      _elementsPerPartialSum() {
      // Allocate the arrays and initialize the data structure.
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGenerator2DSearchStatic(const DiscreteGenerator2DSearchStatic& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _sum(other._sum),
      _partialPmfSums(other._partialPmfSums),
      _elementsPerPartialSum(other._elementsPerPartialSum) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGenerator2DSearchStatic&
   operator=(const DiscreteGenerator2DSearchStatic& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _sum = other._sum;
         _partialPmfSums = other._partialPmfSums;
         _elementsPerPartialSum = other._elementsPerPartialSum;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGenerator2DSearchStatic() {}

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
      // Loop until we get a valid deviate.
      result_type index;
      do {
#ifdef STLIB_DEBUG
         assert(_sum > 0);
#endif
         // A random weighted probability.
         Number r = _continuousUniformGenerator() * _sum;

         // Use the partial PMF sums to step forward. Note that this chop-down
         // search cannot fail because the guard element is the last row, not
         // one past the last row.
         typename std::vector<Number>::const_iterator i =
            _partialPmfSums.begin();
         while (r >= *i) {
            r -= *i;
            ++i;
         }

         // Use a linear search from the offset to finish the search.
         const std::ptrdiff_t offset = _elementsPerPartialSum *
                                       (i - _partialPmfSums.begin());
         index = offset +
                 linearSearchChopDownGuarded(Base::begin() + offset,
                                             Base::end(), r);
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
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      Base::initialize(begin, end);
      // Set the partial sum size to the square root of the array size.
      _elementsPerPartialSum = std::size_t(std::sqrt(double(size())));
      // Allocate the array.  The guard element is the last row.
      _partialPmfSums.resize(size() / _elementsPerPartialSum +
                             (size() % _elementsPerPartialSum != 0));
      // Compute the partial sums.
      repair();
   }

protected:

   //! Set the probability mass function with the specified index.
   /*!
     This is only to be used by derived classes.
   */
   using Base::set;

   //! Repair the data structure.
   /*!
     Compute the sum of the PMF.
   */
   void
   repair() {
      // Recompute the partial sums.
      iterator pmf = Base::begin();
      for (std::size_t i = 0; i != _partialPmfSums.size() - 1; ++i) {
         _partialPmfSums[i] = 0;
         for (std::size_t j = 0; j != _elementsPerPartialSum; ++j) {
            _partialPmfSums[i] += *pmf++;
         }
      }
      // The guard element is the last row (not one past the last row).
      _partialPmfSums.back() = 0.5 * std::numeric_limits<Number>::max();

      // Recompute the total sum.  Use the partial sums and the last row.
      _sum = std::accumulate(_partialPmfSums.begin(),
                             _partialPmfSums.end() - 1, Number(0)) +
             std::accumulate(pmf, Base::end(), Number(0));
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
      out << "PMF sum = " << _sum << "\n"
          << "Elements per partial sum = " << _elementsPerPartialSum << "\n"
          << "Partial sums = \n" << _partialPmfSums << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif

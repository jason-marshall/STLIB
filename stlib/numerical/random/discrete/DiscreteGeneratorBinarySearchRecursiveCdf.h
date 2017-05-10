// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinarySearchRecursiveCdf.h
  \brief discrete deviate.  Binary search.
*/

#if !defined(__numerical_DiscreteGeneratorBinarySearchRecursiveCdf_h__)
#define __numerical_DiscreteGeneratorBinarySearchRecursiveCdf_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ext/vector.h"

#include <boost/config.hpp>

#include <numeric>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

// CONTINUE: Use linear search for a certain number of bits.
//! Discrete deviate generator.  CDF inversion using a partial, recursive CDF.
/*!
  \param Generator The discrete, uniform generator.

  CONTINUE.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinarySearchRecursiveCdf {
   //
   // Public constants.
   //
public:

   //! The sum of the PMF is automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = true;

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
   typedef double Number;
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
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   std::vector<Number> _pmf;
   //! Cumulative distribution function.  (This is scaled and may not approach unity.)
   std::vector<Number> _partialRecursiveCdf;
   //! The error in the sum of the PMF.
   Number _error;
   //! The number of bits needed to index into the _partialRecursiveCdf array.
   std::size_t _indexBits;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinarySearchRecursiveCdf();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinarySearchRecursiveCdf
   (DiscreteUniformGenerator* generator) :
      _continuousUniformGenerator(generator),
      _pmf(),
      _partialRecursiveCdf(),
      _error(0),
      _indexBits(-1) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinarySearchRecursiveCdf(DiscreteUniformGenerator* generator,
         ForwardIterator begin,
         ForwardIterator end) :
      _continuousUniformGenerator(generator),
      _pmf(),
      _partialRecursiveCdf(),
      _error(0),
      _indexBits(-1) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearchRecursiveCdf
   (const DiscreteGeneratorBinarySearchRecursiveCdf& other) :
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _pmf(other._pmf),
      _partialRecursiveCdf(other._partialRecursiveCdf),
      _error(other._error),
      _indexBits(other._indexBits) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearchRecursiveCdf&
   operator=
   (const DiscreteGeneratorBinarySearchRecursiveCdf& other) {
      if (this != &other) {
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _pmf = other._pmf;
         _partialRecursiveCdf = other._partialRecursiveCdf;
         _error = other._error;
         _indexBits = other._indexBits;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinarySearchRecursiveCdf() {}

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
   operator()();

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   Number
   operator[](const std::size_t index) const {
      return _pmf[index];
   }

   //! Get the number of possible deviates.
   std::size_t
   size() const {
      return _pmf.size();
   }

   //! Get the sum of the probability mass functions.
   /*!
     \pre The CDF must be computed before calling this function.
   */
   Number
   sum() const {
      return *(_partialRecursiveCdf.end() - 1);
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      // Recompute the PMF sum if necessary.
      update();
      return sum() > 0;
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
      // Initialize the PMF.
      _pmf.resize(std::distance(begin, end));
      std::copy(begin, end, _pmf.begin());
      // Compute the array size for the partial, recursive CDF.
      std::size_t sz = 1;
      _indexBits = 0;
      while (sz < size()) {
         sz *= 2;
         ++_indexBits;
      }
      _partialRecursiveCdf.resize(sz);

      repair();
   }

   //! Set the probability mass function with the specified index.
   /*!
     Update the partial, recursive CDF using the difference between the new and
     old values.
   */
   void
   set(std::size_t index, Number value) {
      // Update the error in the PMF sum.
      _error += (sum() + value + _pmf[index]) *
                std::numeric_limits<Number>::epsilon();
      // Update the partial, recursive CDF.
      updateCdf(index, value - _pmf[index]);
      // Update the PMF.
      _pmf[index] = value;
   }

   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   set(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != size(); ++i) {
         set(i, iterator[i]);
      }
   }

private:

   //! Check if the data structure needs repair.
   /*!
     This data structure continuously updates the sum of the PMF so we don't
     need to do that here.
   */
   void
   update() {
      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_error > allowedRelativeError * sum()) {
         repair();
      }
   }

   //! Repair the partial, recursive CDF.
   void
   repair();

   //! Update the CDF.
   void
   updateCdf(std::size_t index, const Number difference);

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "Partial, recursive CDF = \n" << _partialRecursiveCdf << '\n'
          << "Index bits = " << _indexBits << "\n";
   }

   //@}
};


} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorBinarySearchRecursiveCdf_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorBinarySearchRecursiveCdf.ipp"
#undef __numerical_random_DiscreteGeneratorBinarySearchRecursiveCdf_ipp__

#endif

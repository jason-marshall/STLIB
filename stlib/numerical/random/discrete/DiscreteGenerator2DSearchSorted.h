// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGenerator2DSearchSorted.h
  \brief discrete deviate.  CDF inversion using a partial sums of the PMF.
*/

#if !defined(__numerical_DiscreteGenerator2DSearchSorted_h__)
#define __numerical_DiscreteGenerator2DSearchSorted_h__

#include <boost/config.hpp>

#include "stlib/numerical/random/discrete/DiscreteGenerator2DSearchSortedStatic.h"

namespace stlib
{
namespace numerical {

//! Discrete deviate.  CDF inversion using 2-D search.
/*!
  \param Generator is the discrete, uniform generator.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGenerator2DSearchSorted :
   public DiscreteGenerator2DSearchSortedStatic<Generator> {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function.
   typedef DiscreteGenerator2DSearchSortedStatic<Generator> Base;

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
   typedef typename Base::Number Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;
   //! The integer type for the rebuild counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //
   // Member data.
   //
protected:

   //! The sum of the PMF.
   using Base::_sum;
   //! Partial sums of the PMF.
   using Base::_partialPmfSums;
   //! The elements per partial sum.
   using Base::_elementsPerPartialSum;

   //! The row indices for the PMF array.
   std::vector<std::size_t> _row;
   //! The error in the sum of the PMF.
   Number _error;
   //! The number of times you can set a PMF element between rebuilds.
   ads::CounterWithReset<> _rebuildCounter;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGenerator2DSearchSorted();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGenerator2DSearchSorted(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(generator),
      _row(),
      _error(0),
      _rebuildCounter(Counter(0)) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGenerator2DSearchSorted(DiscreteUniformGenerator* generator,
                                   ForwardIterator begin, ForwardIterator end) :
      Base(generator),
      _row(),
      _error(0),
      _rebuildCounter(Counter(0)) {
      // Allocate the arrays and initialize the data structure.
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGenerator2DSearchSorted(const DiscreteGenerator2DSearchSorted& other) :
      Base(other),
      _row(other._row),
      _error(other._error),
      _rebuildCounter(other._rebuildCounter) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGenerator2DSearchSorted&
   operator=(const DiscreteGenerator2DSearchSorted& other) {
      if (this != &other) {
         Base::operator=(other);
         _row = other._row;
         _error = other._error;
         _rebuildCounter = other._rebuildCounter;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGenerator2DSearchSorted() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   using Base::seed;

   //! Return a discrete deviate.
   result_type
   operator()() {
      return Base::operator()();
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
   using Base::sum;

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      // Recompute the PMF sum if necessary.
      update();
      return Base::isValid();
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

      // Set the row indices for each element.
      _row.resize(size());
      std::vector<std::size_t>::iterator i = _row.begin();
      for (std::size_t row = 0; row != _partialPmfSums.size(); ++row) {
         for (std::size_t col = 0;
               col != _elementsPerPartialSum && i != _row.end(); ++col) {
            *i++ = row;
         }
      }

      // Set an appropriate number of times between rebuilds.
      _rebuildCounter.setReset(std::max(std::size_t(1000), size()));
   }

   //! Set the probability mass function with the specified index.
   /*!
     Update the partial sums and the total sum of the PMF using the difference
     between the new and old values.
   */
   void
   set(std::size_t index, Number value) {
      // Update the error in the PMF sum.
      _error += (_sum + value + operator[](index)) *
                std::numeric_limits<Number>::epsilon();
      // Update the PMF sum with the difference between the new and old values.
      const Number difference = value - operator[](index);
      _sum += difference;
      // Update the row sum.
      _partialPmfSums[_row[Base::position(index)]] += difference;
      // Set the PMF value.
      Base::set(index, value);
   }

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter n) {
      _rebuildCounter.setReset(n);
   }

private:

   //! Check if the data structure needs repair.
   void
   update() {
      // Do the rebuilding first as rebuilding also repairs.
      // Resort if necessary.
      if (_rebuildCounter() <= 0) {
         rebuild();
         return;
      }

      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_error > allowedRelativeError * _sum) {
         repair();
      }
   }

   //! Repair the data structure.
   /*!
     Recompute the sum of the PMF.
   */
   void
   repair() {
      Base::repair();
      // The initial error in the sum.
      _error = size() * _sum * std::numeric_limits<Number>::epsilon();
   }

   //! Sort the PMF.
   void
   rebuild() {
      // The base class rebuilds and repairs.
      Base::rebuild();
      // The initial error in the sum.
      _error = size() * _sum * std::numeric_limits<Number>::epsilon();
      _rebuildCounter.reset();
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
      out << "Row indices = \n" << _row << '\n'
          << "Error in the PMF sum = " << _error << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif

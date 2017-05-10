// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGenerator2DSearchBubbleSort.h
  \brief discrete deviate.  CDF inversion using a partial sums of the PMF.
*/

#if !defined(__numerical_DiscreteGenerator2DSearchBubbleSort_h__)
#define __numerical_DiscreteGenerator2DSearchBubbleSort_h__

#include "stlib/numerical/random/discrete/DgPmfOrderedPairPointer.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ads/counter/CounterWithReset.h"

#include <boost/config.hpp>

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  CDF inversion using 2-D search.
/*!
  \param Generator is the discrete, uniform generator.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGenerator2DSearchBubbleSort :
   public DgPmfOrderedPairPointer<true> {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function.
   typedef DgPmfOrderedPairPointer<true> Base;

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

   //
   // Member data.
   //
protected:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The last deviate drawn.
   mutable result_type _deviate;
   //! The row indices for the PMF array.
   std::vector<std::size_t> _row;
   //! The predecessor indices.
   std::vector<std::size_t> _predecessors;
   //! The sum of the PMF.
   Number _sum;
   //! The error in the sum of the PMF.
   Number _error;
   //! Partial sums of the PMF.
   std::vector<Number> _partialPmfSums;
   //! The elements per partial sum.
   std::size_t _elementsPerPartialSum;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGenerator2DSearchBubbleSort();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGenerator2DSearchBubbleSort(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _deviate(0),
      _row(),
      _predecessors(),
      _sum(0),
      _error(0),
      // Empty array.
      _partialPmfSums(),
      _elementsPerPartialSum(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGenerator2DSearchBubbleSort(DiscreteUniformGenerator* generator,
                                       ForwardIterator begin, ForwardIterator end) :
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _deviate(0),
      _row(),
      _predecessors(),
      _sum(0),
      _error(0),
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
   DiscreteGenerator2DSearchBubbleSort(const DiscreteGenerator2DSearchBubbleSort& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _deviate(other._deviate),
      _row(other._row),
      _predecessors(other._predecessors),
      _sum(other._sum),
      _error(other._error),
      _partialPmfSums(other._partialPmfSums),
      _elementsPerPartialSum(other._elementsPerPartialSum) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGenerator2DSearchBubbleSort&
   operator=(const DiscreteGenerator2DSearchBubbleSort& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _deviate = other._deviate;
         _row = other._row;
         _predecessors = other._predecessors;
         _sum = other._sum;
         _error = other._error;
         _partialPmfSums = other._partialPmfSums;
         _elementsPerPartialSum = other._elementsPerPartialSum;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGenerator2DSearchBubbleSort() {}

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
      do {
         // A random weighted probability.
         Number r = _continuousUniformGenerator() * _sum;

         // Use the partial PMF sums to step forward. Note that this chop-down
         // search cannot fail because the guard element is the last row, not
         // one past the last row.
         typename std::vector<Number>::const_iterator i = _partialPmfSums.begin();
         while (r >= *i) {
            r -= *i;
            ++i;
         }

         // Use a linear search from the offset to finish the search.
         const std::ptrdiff_t offset = _elementsPerPartialSum *
                                       (i - _partialPmfSums.begin());
         _deviate = linearSearchChopDownGuardedPair(Base::begin() + offset,
                    Base::end(), r);
      }
      while (operator[](_deviate) == 0);
      return _deviate;
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
   //! Return the position of the specified event in the ordered array.
   using Base::position;

   //! Get the sum of the probability mass functions.
   Number
   sum() const {
      return _sum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      // Recompute the PMF sum if necessary.
      update();
      return _sum > 0;
   }

   //! Return the expected cost for generating a deviate.
   Number
   cost() const {
      Number c = 0;
      for (std::size_t i = 0; i != size(); ++i) {
         std::size_t row = _row[i];
         std::size_t col = i - row * _elementsPerPartialSum;
         c += (row + col + 2) * _pmfPairs[i].first;
      }
      return c / sum();
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

      // Set the row indices for each element.
      _row.resize(size());
      std::vector<std::size_t>::iterator i = _row.begin();
      for (std::size_t row = 0; row != _partialPmfSums.size(); ++row) {
         for (std::size_t col = 0;
               col != _elementsPerPartialSum && i != _row.end(); ++col) {
            *i++ = row;
         }
      }

      // Compute the predecessors.
      // The element indices for 9 elements is shown below.
      // 6 7 8
      // 3 4 5
      // 0 1 2
      // The ordering is:
      // 5 7 8
      // 2 4 6
      // 0 1 3
      // For this case the predecessors are:
      // 4 5 7
      // 1 2 6
      // 9 0 3
      // Here 9 indicates no predecessor.
      _predecessors.resize(size());
      // The first element has no predecessor.
      _predecessors[0] = size();
      std::size_t row = 0, col = 0;
      std::size_t predecessor;
      for (std::size_t i = 1; i != size(); ++i) {
         // Record the predecessor.
         predecessor = row * _elementsPerPartialSum + col;
         // Move forward.
         if (col == 0) {
            col = row + 1;
            row = 0;
         }
         else {
            ++row;
            --col;
         }
         // Correct for running out of the square.
         if (col >= _elementsPerPartialSum) {
            std::size_t d = col - _elementsPerPartialSum + 1;
            col -= d;
            row += d;
         }
         // Correct for running past the number of elements.
         if (row* _elementsPerPartialSum + col >= size()) {
            col = row + col + 1;
            row = 0;
            // Correct for running out of the square.
            if (col >= _elementsPerPartialSum) {
               std::size_t d = col - _elementsPerPartialSum + 1;
               col -= d;
               row += d;
            }
         }
         // Set the predecessor.
         _predecessors[row* _elementsPerPartialSum + col] = predecessor;
      }

      // Valid value for the last deviate drawn.
      _deviate = 0;
      // Sort the PMF.
      sort();
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

private:

   //! Check if the data structure needs repair.
   void
   update() {
      //
      // Use swapping to perhaps move the last event to a position with smaller
      // metropolis distance from the lower corner of the 2-D array.
      //
      swap();

      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_error > allowedRelativeError * _sum) {
         repair();
      }
   }

   void
   swap() {
      const std::size_t position = Base::position(_deviate);
      // If we are at the lower corner do nothing.
      if (position == 0) {
         return;
      }
      const std::size_t row = _row[position];
      const std::size_t col = position - row * _elementsPerPartialSum;
      // First try to move within the row so we don't have to modify the
      // partial sums.
      if (col != 0) {
         if (_pmfPairs[position - 1].first < _pmfPairs[position].first) {
            // Swap the pointers
            std::swap(_pointers[_pmfPairs[position-1].second],
                      _pointers[_pmfPairs[position].second]);
            // Swap the value/index pairs.
            std::swap(_pmfPairs[position-1], _pmfPairs[position]);
            return;
         }
      }
      // Otherwise try moving to the predecessor.
      const std::size_t p = _predecessors[position];
      const Number difference = _pmfPairs[p].first -
                                _pmfPairs[position].first;
      if (difference < 0) {
         const std::size_t r = _row[p];
         // Update the partial sums.
         _partialPmfSums[row] += difference;
         _partialPmfSums[r] -= difference;
         swapPositions(position, p);
      }
   }

   void
   swapOld() {
      const std::size_t position = Base::position(_deviate);
      // If we are not at the lower corner.
      if (position != 0) {
         const std::size_t row = _row[position];
         const std::size_t col = position - row * _elementsPerPartialSum;
         // First try to move within the row so we don't have to modify the
         // partial sums.
         if (col != 0) {
            if (_pmfPairs[position - 1].first < _pmfPairs[position].first) {
               // Swap the pointers
               std::swap(_pointers[_pmfPairs[position-1].second],
                         _pointers[_pmfPairs[position].second]);
               // Swap the value/index pairs.
               std::swap(_pmfPairs[position-1], _pmfPairs[position]);
               return;
            }
         }
         // We are in the first column. Try swapping with an element that is close
         // to the diagonal.
         else {
            const std::size_t r = (row - 1) / 2;
            const std::size_t c = row / 2;
            const std::size_t p = r * _elementsPerPartialSum + c;
            const Number difference = _pmfPairs[p].first -
                                      _pmfPairs[position].first;
            if (difference < 0) {
               // Update the partial sums.
               _partialPmfSums[row] += difference;
               _partialPmfSums[r] -= difference;
               swapPositions(position, p);
               return;
            }
         }
         // Otherwise try to move down a row.
         if (row != 0) {
            const std::size_t r = row - 1;
            const std::size_t p = position - _elementsPerPartialSum;
            const Number difference = _pmfPairs[p].first -
                                      _pmfPairs[position].first;
            if (difference < 0) {
               // Update the partial sums.
               _partialPmfSums[row] += difference;
               _partialPmfSums[r] -= difference;
               swapPositions(position, p);
               return;
            }
         }
         // We are in the first row. Try swapping with an element that is close
         // to the diagonal.
         else {
            const std::size_t r = col / 2;
            const std::size_t c = (col - 1) / 2;
            const std::size_t p = r * _elementsPerPartialSum + c;
            const Number difference = _pmfPairs[p].first -
                                      _pmfPairs[position].first;
            if (difference < 0) {
               // Update the partial sums.
               _partialPmfSums[row] += difference;
               _partialPmfSums[r] -= difference;
               swapPositions(position, p);
               return;
            }
         }
      }
   }

   void
   updatePartialSums(const std::size_t /*source*/,
                     const std::size_t /*destination*/) {
   }

   //! Swap the two elements in the PMF array.
   void
   swapPositions(const std::size_t i, const std::size_t j) {
      // Swap the pointers
      std::swap(_pointers[_pmfPairs[i].second], _pointers[_pmfPairs[j].second]);
      // Swap the value/index pairs.
      std::swap(_pmfPairs[i], _pmfPairs[j]);
   }

   //! Repair the data structure.
   /*!
     Recompute the sum of the PMF.
   */
   void
   repair() {
      // Recompute the partial sums.
      iterator pmf = Base::begin();
      for (std::size_t i = 0; i != _partialPmfSums.size() - 1; ++i) {
         _partialPmfSums[i] = 0;
         for (std::size_t j = 0; j != _elementsPerPartialSum; ++j) {
            _partialPmfSums[i] += pmf->first;
            ++pmf;
         }
      }
      // The guard element is the last row (not one past the last row).
      _partialPmfSums.back() = 0.5 * std::numeric_limits<Number>::max();

      // Recompute the total sum.  Use the partial sums and the last row.
      _sum = std::accumulate(_partialPmfSums.begin(),
                             _partialPmfSums.end() - 1, Number(0));
      for (; pmf != Base::end(); ++pmf) {
         _sum += pmf->first;
      }
      // The initial error in the sum.
      _error = size() * _sum * std::numeric_limits<Number>::epsilon();
   }

   //! Sort the PMF.
   void
   sort() {
      // First sort the PMF.
      std::sort(Base::begin(), Base::end(), Base::ValueGreater());
      // Copy the value/index pairs.
      Base::Container sorted(Base::_pmfPairs);
      // Arrange using the metropolis distance.
      // We can split the 2-D array into three regions: lower triangle, upper
      // triangle, and last row. These are indicated with numbers below.
      //
      // 3 3
      // 1 1 2 2
      // 1 1 1 2
      // 1 1 1 1
      //
      // Note that the last row may not be full.
      Base::const_iterator s = sorted.begin();
      std::size_t start;
      // Lower triangle.
      for (start = 0; start != _elementsPerPartialSum; ++start) {
         // Number of elements along the diagonal.
         std::size_t length = std::min(start + 1, _partialPmfSums.size() - 1);
         for (std::size_t i = 0; i != length; ++i) {
            Base::_pmfPairs[start + i *(_elementsPerPartialSum - 1)] = *s++;
         }
      }
      // Upper triangle.
      for (std::size_t row = 1; row < _partialPmfSums.size() - 1; ++row) {
         start = (row + 1) * _elementsPerPartialSum - 1;
         for (std::size_t i = 0; i < _partialPmfSums.size() - row - 1; ++i) {
            Base::_pmfPairs[start + i *(_elementsPerPartialSum - 1)] = *s++;
         }
      }
      // Last row.
      for (start = (_partialPmfSums.size() - 1) * _elementsPerPartialSum;
            start != _pmfPairs.size(); ++start) {
         Base::_pmfPairs[start] = *s++;
      }
#ifdef STLIB_DEBUG
      assert(s == sorted.end());
#endif

      Base::computePointers();

      // Recompute the PMF sum and row sums.
      repair();
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
      {
         std::cout << "2-D array of values:\n";
         const_iterator i = Base::begin();
         for (std::size_t row = 0; row != _partialPmfSums.size(); ++row) {
            for (std::size_t col = 0;
                  col != _elementsPerPartialSum && i != Base::end(); ++col) {
               std::cout << i->first << ' ';
               ++i;
            }
            std::cout << '\n';
         }
      }
      {
         std::cout << "Predecessors:\n";
         std::vector<std::size_t>::const_iterator i = _predecessors.begin();
         for (std::size_t row = 0; row != _partialPmfSums.size(); ++row) {
            for (std::size_t col = 0;
                  col != _elementsPerPartialSum && i != _predecessors.end(); ++col) {
               std::cout << *i << ' ';
               ++i;
            }
            std::cout << '\n';
         }
      }
      out << "Row indices = \n" << _row << '\n'
          << "PMF sum = " << _sum << "\n"
          << "Error in the PMF sum = " << _error << "\n"
          << "Elements per partial sum = " << _elementsPerPartialSum << "\n"
          << "Partial sums = \n" << _partialPmfSums << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif

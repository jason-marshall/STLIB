// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorCompositionRejectionStatic.h
  \brief Discrete deviate.  Composition rejection.
*/

#if !defined(__numerical_DiscreteGeneratorCompositionRejectionStatic_h__)
#define __numerical_DiscreteGeneratorCompositionRejectionStatic_h__

#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ext/vector.h"

#include <boost/config.hpp>

#include <tuple>

#include <algorithm>
#include <deque>
#include <numeric>

#include <cmath>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! A group in the composition-rejection method.
class DgcrGroup {
   //
   // Member data.
   //
private:

   std::vector<std::size_t> _indices;
   int _exponent;
   // 2**_exponent.
   double _upperBound;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Default constructor. Make an invalid group.
   DgcrGroup() :
      // Empty.
      _indices(),
      // Invalid.
      _exponent(std::numeric_limits<int>::min()),
      _upperBound(0) {}

   //! Construct from the exponent.
   DgcrGroup(const int e) :
      // Empty.
      _indices(),
      _exponent(e),
      _upperBound(std::ldexp(1., e)) {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{

   //! Return the exponent.
   int
   exponent() const {
      return _exponent;
   }

   //! Return an upper bound on the probabilities in this group.
   double upperBound() const {
      return _upperBound;
   }

   //! Return true if the container is empty.
   bool
   empty() const {
      return _indices.empty();
   }

   //! Return the number of elements.
   std::size_t
   size() const {
      return _indices.size();
   }

   std::size_t
   operator[](const std::size_t i) const {
      return _indices[i];
   }

   std::size_t
   back() const {
      return _indices.back();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{

   void
   setExponent(const int e) {
      _exponent = e;
      _upperBound = std::ldexp(1., e);
   }

   //! Insert by adding to the back.
   void
   insert(const std::size_t i) {
      _indices.push_back(i);
   }

   //! Erase by moving the last element to the specified location.
   void
   erase(const std::size_t i) {
      _indices[i] = _indices.back();
      _indices.pop_back();
   }

   //@}
};


//! Discrete deviate.  Composition rejection.
/*!
  \param Generator is the discrete, uniform generator.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorCompositionRejectionStatic {
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
   // Private Types.
   //
private:

   typedef std::tuple<Number, unsigned, unsigned> Tuple;
   typedef std::vector<Tuple> VectorOfTuple;

   //
   // Constants.
   //
protected:

   //! The maximum allowed number of leading empty groups.
   BOOST_STATIC_CONSTEXPR std::size_t MaxEmpty = 10;
   //! The number of groups.
   BOOST_STATIC_CONSTEXPR std::size_t NumberOfGroups = 32 + MaxEmpty + 1;

   //
   // Member data.
   //
protected:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;
   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The PMF and indices.
   VectorOfTuple _pmf;
   //! The groups.
   std::deque<DgcrGroup> _groups;
   //! Group sums of the PMF.
   std::vector<Number> _groupSums;
   //! The sum of the PMF.
   Number _sum;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorCompositionRejectionStatic();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorCompositionRejectionStatic
   (DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      // Empty.
      _pmf(),
      _groups(),
      _groupSums(NumberOfGroups + 1, 0.),
      _sum(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorCompositionRejectionStatic
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      _discreteUniformGenerator(generator),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      // Empty.
      _pmf(),
      _groups(),
      _groupSums(NumberOfGroups + 1, 0.),
      _sum(0) {
      // Allocate the arrays and initialize the data structure.
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorCompositionRejectionStatic
   (const DiscreteGeneratorCompositionRejectionStatic& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _pmf(other._pmf),
      _groups(other._groups),
      _groupSums(other._groupSums),
      _sum(other._sum) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorCompositionRejectionStatic&
   operator=(const DiscreteGeneratorCompositionRejectionStatic& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _pmf = other._pmf;
         _groups = other._groups;
         _groupSums = other._groupSums;
         _sum = other._sum;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorCompositionRejectionStatic() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a discrete deviate.
   result_type
   operator()() {
      // Loop until we get a valid group.
      std::size_t groupIndex;
      do {
         groupIndex =
            linearSearchChopDownGuarded(_groupSums.begin(), _groupSums.end(),
                                        _continuousUniformGenerator() * _sum);
      }
      while (_groups[groupIndex].empty());
      // Above I check that the group is not empty instead of checking that
      // the group sum is nonzero. Round-off error might cause the latter test
      // to fail.
#ifdef STLIB_DEBUG
      assert(groupIndex < NumberOfGroups);
#endif
      const DgcrGroup& group = _groups[groupIndex];

      // Use the rejection method to get a deviate.
      result_type index;
      unsigned deviate, offset;
      do {
         // Select a bin by generating a uniform discrete deviate in the range
         // [0..group.size()).
         do {
            deviate = (*_discreteUniformGenerator)();
            offset = (deviate / group.size()) * group.size();
         }
         while (offset > std::numeric_limits<unsigned>::max() - group.size());
         index = group[deviate - offset];
      }
      while (_continuousUniformGenerator() * group.upperBound() >
             operator[](index));
      return index;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability with the specified index.
   Number
   operator[](const std::size_t i) const {
      return std::get<0>(_pmf[i]);
   }

   //! Get the number of possible deviates.
   std::size_t
   size() const {
      return _pmf.size();
   }

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

protected:

   //! Return the group for the probability.
   /*!
     The last group has no lower bound. A negative index for the group
     indicates that new groups need to be added.
   */
   int
   computeGroup(const Number x) const {
      // Check for the last (overflow) group.
      if (x < _groups.back().upperBound()) {
         return NumberOfGroups;
      }
      int e;
      std::frexp(x, &e);
      return _groups[0].exponent() - e;
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
      // Copy the probabilities.
      _pmf.resize(std::distance(begin, end));
      const Number upperBound = *std::max_element(begin, end);
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         std::get<0>(_pmf[i]) = *begin++;
      }

      _groups.clear();
      int exponent;
      std::frexp(upperBound, &exponent);
      // Include the overflow group.
      for (std::size_t i = 0; i != NumberOfGroups + 1; ++i) {
         _groups.push_back(DgcrGroup(exponent--));
      }

      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         const int index = computeGroup(std::get<0>(_pmf[i]));
#ifdef STLIB_DEBUG
         assert(index >= 0);
#endif
         std::get<1>(_pmf[i]) = index;
         std::get<2>(_pmf[i]) = _groups[index].size();
         _groups[index].insert(i);
      }

      // Compute the group sums.
      for (std::size_t i = 0; i != NumberOfGroups; ++i) {
         repair(i);
      }
      // Compute the sum of the PMF.
      repair();
   }

protected:

   //! Repair the sum for the specified group.
   void
   repair(const std::size_t index) {
      const DgcrGroup& group = _groups[index];
      Number sum = 0;
      for (std::size_t i = 0; i != group.size(); ++i) {
         sum += std::get<0>(_pmf[group[i]]);
      }
      _groupSums[index] = sum;
   }

   //! Recompute the sum of the PMF.
   void
   repair() {
      // Set the guard element.
      _groupSums.back() = 0.5 * std::numeric_limits<Number>::max();
      _sum = std::accumulate(_groupSums.begin(),
                             _groupSums.begin() + NumberOfGroups, Number(0));
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "Probability, group, and position:\n";
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         out << std::get<0>(_pmf[i]) << ' '
             << std::get<1>(_pmf[i]) << ' '
             << std::get<2>(_pmf[i]) << '\n';
      }
      out << "Groups:\n";
      for (std::size_t i = 0; i != _groups.size(); ++i) {
         out << i << '\n';
         for (std::size_t j = 0; j != _groups[i].size(); ++j) {
            out << _groups[i][j] << ' ';
         }
         out << '\n';
         out << "exponent = " << _groups[i].exponent() << ' '
             << "upperBound = " << _groups[i].upperBound() << '\n';
      }
      out << "Group sums = " << _groupSums << '\n'
          << "PMF sum = " << _sum << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif

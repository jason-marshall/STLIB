// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorCompositionRejection.h
  \brief Discrete deviate.  Composition rejection.
*/

#if !defined(__numerical_DiscreteGeneratorCompositionRejection_h__)
#define __numerical_DiscreteGeneratorCompositionRejection_h__

#include "stlib/numerical/random/discrete/DiscreteGeneratorCompositionRejectionStatic.h"

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Composition rejection.
/*!
  \param Generator is the discrete, uniform generator.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorCompositionRejection :
   public DiscreteGeneratorCompositionRejectionStatic<Generator> {
   //
   // Private types.
   //
private:

   //! Composition rejection for a static PMF.
   typedef DiscreteGeneratorCompositionRejectionStatic<Generator> Base;

   //
   // Constants.
   //
private:

   using Base::MaxEmpty;
   using Base::NumberOfGroups;

   //
   // Constants.
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

   //! The PMF and indices.
   using Base::_pmf;
   //! The groups.
   using Base::_groups;
   //! Group sums of the PMF.
   using Base::_groupSums;
   //! The sum of the PMF.
   using Base::_sum;

   //! The error in the sum of the PMF.
   Number _sumError;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorCompositionRejection();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorCompositionRejection(DiscreteUniformGenerator* generator) :
      Base(generator),
      _sumError(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorCompositionRejection(DiscreteUniformGenerator* generator,
                                         ForwardIterator begin,
                                         ForwardIterator end) :
      Base(generator),
      _sumError(0) {
      // Allocate the arrays and initialize the data structure.
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorCompositionRejection
   (const DiscreteGeneratorCompositionRejection& other) :
      Base(other),
      _sumError(other._sumError) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorCompositionRejection&
   operator=(const DiscreteGeneratorCompositionRejection& other) {
      if (this != &other) {
         Base::operator=(other);
         _sumError = other._sumError;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorCompositionRejection() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   using Base::seed;

   //! Return a discrete deviate.
   using Base::operator();

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
      // Compute the error in the sum of the PMF.
      _sumError = size() * sum() * std::numeric_limits<Number>::epsilon();
   }

   //! Set the probability mass function with the specified index.
   /*!
     Update the partial sums and the total sum of the PMF using the difference
     between the new and old values.
   */
   void
   set(const std::size_t index, const Number value) {
      // Update the error in the PMF sum.
      _sumError += (_sum + value + operator[](index)) *
                   std::numeric_limits<Number>::epsilon();
      // Update the PMF sum with the difference between the new and old values.
      const Number difference = value - operator[](index);
      _sum += difference;

      int newGroup;
      while ((newGroup = Base::computeGroup(value)) < 0) {
         insertGroup();
      }
      const int oldGroup = std::get<1>(_pmf[index]);
      // If the probability remains in the same group.
      if (newGroup == oldGroup) {
         _groupSums[oldGroup] += difference;
         // Update PMF.
         std::get<0>(_pmf[index]) = value;
      }
      // The probability changes groups.
      else {
         // Remove from the old group.
         _groupSums[oldGroup] -= operator[](index);
         unsigned position = std::get<2>(_pmf[index]);
         std::get<1>(_pmf[_groups[oldGroup].back()]) = oldGroup;
         std::get<2>(_pmf[_groups[oldGroup].back()]) = position;
         _groups[oldGroup].erase(position);
         // Add to the new group.
         _groupSums[newGroup] += value;
         std::get<1>(_pmf[index]) = newGroup;
         std::get<2>(_pmf[index]) = _groups[newGroup].size();
         _groups[newGroup].insert(index);

         // Update PMF.
         std::get<0>(_pmf[index]) = value;
      }
   }

private:

   //! Add a new group to the front of the deque.
   void
   insertGroup() {
      // Update the group sums.
      for (std::size_t i = NumberOfGroups - 1; i != 0; --i) {
         _groupSums[i] = _groupSums[i-1];
      }
      _groupSums[0] = 0;
      // Update the group indices.
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         ++std::get<1>(_pmf[i]);
      }
      // The last valid group becomes the "effectively zero" group.
      // Combine the last valid group with the "effectively zero" group.
      DgcrGroup& lastValid = _groups[NumberOfGroups-1];
      DgcrGroup& zero = _groups.back();
      for (std::size_t i = 0; i != zero.size(); ++i) {
         std::get<1>(_pmf[zero[i]]) = NumberOfGroups;
         std::get<2>(_pmf[zero[i]]) = lastValid.size();
         lastValid.insert(zero[i]);
      }
      _groups.pop_back();
      // Add a new group to the front.
      _groups.push_front(DgcrGroup(_groups.front().exponent() + 1));
   }

   //! Erase an empty group from the front of the deque.
   void
   eraseGroup() {
      // Update the group sums.
      for (std::size_t i = 0; i != NumberOfGroups - 1; ++i) {
         _groupSums[i] = _groupSums[i+1];
      }
      // Erase the empty group.
      _groups.pop_front();
      // Append the new "effectively zero" group.
      _groups.push_back(DgcrGroup(_groups.back().exponent() - 1));
      // Update the group indices.
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         --std::get<1>(_pmf[i]);
      }
      // Remove the effectively zero probabilities from the last valid group.
      DgcrGroup& lastValid = _groups[NumberOfGroups-1];
      DgcrGroup& zero = _groups.back();
      for (std::size_t i = 0; i != lastValid.size(); /*No increment here.*/) {
         // If the probability is effectively zero.
         const std::size_t index = lastValid[i];
         if (std::get<0>(_pmf[index]) < zero.upperBound()) {
            std::get<1>(_pmf[index]) = NumberOfGroups;
            std::get<2>(_pmf[index]) = zero.size();
            lastValid.erase(i);
            zero.insert(index);
         }
         else {
            ++i;
         }
      }
      // Repair the sum for the last valid group.
      repair(NumberOfGroups - 1);
   }

   //! Check if the data structure needs repair.
   void
   update() {
      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_sumError > allowedRelativeError * sum()) {
         repair();
      }
      //
      // Erase leading groups if necessary.
      //
      // If the sum of the probabilities is zero, all of the groups are empty.
      // We should not erase groups in this case.
      if (sum() != 0) {
         // Find the first non-empty group.
         std::size_t firstNonEmpty = 0;
         while (firstNonEmpty != NumberOfGroups &&
                _groups[firstNonEmpty].empty()) {
            ++firstNonEmpty;
         }
         // Erase as many leading groups as necessary.
         while (firstNonEmpty > MaxEmpty) {
            // Erase the first group.
            --firstNonEmpty;
            eraseGroup();
         }
      }
   }

   //! Repair the sum for the specified group.
   void
   repair(const std::size_t index) {
      // The group sum.
      const DgcrGroup& group = _groups[index];
      Number groupSum = 0;
      for (std::size_t i = 0; i != group.size(); ++i) {
         groupSum += std::get<0>(_pmf[group[i]]);
      }
      // Update the PMF sum and its error.
      _sumError += (_sum + groupSum + _groupSums[index]) *
                   std::numeric_limits<Number>::epsilon();
      _sum += groupSum - _groupSums[index];
      // The group sum and its initial error.
      _groupSums[index] = groupSum;
   }

   //! Repair the group sums and the overall sum.
   /*!
     Recompute the sum of the PMF.
   */
   void
   repair() {
      // Recompute the group sums and their errors.
      for (std::size_t i = 0; i != NumberOfGroups; ++i) {
         // The group sum.
         const DgcrGroup& group = _groups[i];
         _groupSums[i] = 0;
         for (std::size_t j = 0; j != group.size(); ++j) {
            _groupSums[i] += std::get<0>(_pmf[group[j]]);
         }
      }
      // The PMF sum.
      Base::repair();
      // The initial error in the sum.
      _sumError = size() * sum() * std::numeric_limits<Number>::epsilon();
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
      out << "PMF sum error = " << _sumError << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif

// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfIntegerNoBranching.h
  \brief Probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmfIntegerNoBranching_h__)
#define __numerical_DgPmfIntegerNoBranching_h__

#include "stlib/numerical/random/discrete/DgPmfAndSum.h"
#include "stlib/numerical/random/discrete/DgPmf.h"

#include "stlib/ext/vector.h"

#include <limits>

namespace stlib
{
namespace numerical {

//! Whether to use branching and meta-programming.
template<bool _IsDynamic, typename _Integer>
class TraitsForDynamicAndInteger {
public:
   //! Whether it is dynamic.
   BOOST_STATIC_CONSTEXPR bool IsDynamic = _IsDynamic;
   //! The integer type.
   typedef _Integer Integer;
};

//! Probability mass function for a discrete generator.
/*!
  Manage the probability mass function.
*/
template < class PmfAndSum = DgPmfAndSum<DgPmf<> >,
         class Traits = TraitsForDynamicAndBranchingAndInteger < false /*CONTINUE*/, true, int > >
class DgPmfIntegerNoBranching;


//! Probability mass function for a discrete generator.
/*!
  Manage the probability mass function.  Use an array of signed integers
  that represent the probabilities with fixed precision.

  \note  This is just a test.  It does not yet have dynamic capability.
*/
template<class PmfAndSum, typename _Integer>
class DgPmfIntegerNoBranching<PmfAndSum, TraitsForDynamicAndInteger<false, _Integer> > :
   public PmfAndSum {
   //
   // Private types.
   //
private:

   //! The PMF and sum base.
   typedef PmfAndSum Base;

   //
   // Public types.
   //
public:

   //! The floating point number type.
   typedef typename Base::Number Number;
   //! The integer type for the repair counter.
   typedef typename Base::Counter Counter;

   //
   // More private types.
   //
private:

   //! The traits.
   typedef TraitsForDynamicAndInteger<false, _Integer> Traits;
   //! The integer type.
   typedef typename Traits::Integer Integer;

   //! The array type for floating point numbers.
   typedef std::vector<Number> NumberContainer;
   //! The array type for integers.
   typedef std::vector<Integer> IntegerContainer;

   //
   // More public types.
   //
public:

   //! An iterator on the probabilities.
   typedef typename NumberContainer::iterator Iterator;
   //! A const iterator on the probabilities.
   typedef typename NumberContainer::const_iterator ConstIterator;

   //
   // Member data.
   //
private:

   //! Fixed precision probability mass function.
   IntegerContainer _ipmf;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfIntegerNoBranching() :
      Base(),
      // The PMF array is empty.
      _ipmf() {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DgPmfIntegerNoBranching(ForwardIterator begin, ForwardIterator end) :
      Base(),
      _ipmf() {
      initialize(begin, end);
   }

   //! Copy constructor.
   DgPmfIntegerNoBranching(const DgPmfIntegerNoBranching& other) :
      Base(other),
      _ipmf(other._ipmf) {}

   //! Assignment operator.
   DgPmfIntegerNoBranching&
   operator=(const DgPmfIntegerNoBranching& other) {
      if (this != &other) {
         Base::operator=(other);
         _ipmf = other._ipmf;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfIntegerNoBranching() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
protected:

   //! Return a discrete deviate.
   /*!
     Use a linear search to sum probabilities until the sum reaches r.
     For this version, you specify an offset when you have allready subtracted
     a number of the probabilities.
   */
   std::size_t
   operator()(const unsigned r, const std::size_t offset) const {
      return Base::getIndex
             (offset +
              linearSearchChopDownNoBranching(_ipmf.begin() + offset, _ipmf.end(),
                                              r));
   }

   //! Return a discrete deviate.
   /*!
     Use a linear search to sum probabilities until the sum reaches r.
   */
   std::size_t
   operator()(const unsigned r) const {
      return Base::getIndex(linearSearchChopDownNoBranching(_ipmf.begin(),
                            _ipmf.end(), r));
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   using Base::getPmf;

   //! Get the number of possible deviates.
   using Base::getSize;

   //! Get the sum of the probability mass functions.
   using Base::getPmfSum;

   //! Return true if the sum of the PMF is positive.
   using Base::isValid;

   //! Get the number of steps between repairs.
   using Base::getStepsBetweenRepairs;

protected:

   //! Get the beginning of the probabilities in the PMF.
   using Base::getPmfBeginning;

   //! Get the end of the probabilities in the PMF.
   using Base::getPmfEnd;

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgPmfIntegerNoBranching& other) const {
      return Base::operator==(other) && _ipmf == other._ipmf;
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
      // Initialize the floating point PMF and the sum.
      Base::initialize(begin, end);

      //
      // Initialize the fixed precision PMF.
      // The integer array will sum to exactly
      // std::numeric_limits<Integer>::max().
      //

      // Resize the array.
      _ipmf.resize(getSize());
      Number sum = getPmfSum();
      Integer iSum = std::numeric_limits<Integer>::max();
      for (std::size_t i = 0; i < _ipmf.size() - 1; ++i) {
         // Note: I use getPmfBeginning() instead of getPmf() because the
         // array may be sorted.
         _ipmf[i] = Integer(Number(iSum) * getPmfBeginning()[i] / sum);
         sum -= getPmfBeginning()[i];
         iSum -= _ipmf[i];
      }
      // Set the last element.
      *(_ipmf.end() - 1) = iSum;
   }

   //! Set the number of steps between repairs.
   using Base::setStepsBetweenRepairs;

protected:

   //! Get the beginning of the probabilities in the PMF.
   //using Base::getPmfBeginning;

   //! Get the end of the probabilities in the PMF.
   //using Base::getPmfEnd;

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      Base::print(out);
      out << "Integer PMF = \n" << _ipmf << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif

// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfInteger.h
  \brief Probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmfInteger_h__)
#define __numerical_DgPmfInteger_h__

#include "stlib/numerical/random/discrete/DgPmfAndSum.h"
#include "stlib/numerical/random/discrete/DgPmf.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/ext/vector.h"

#include <boost/config.hpp>

#include <limits>

namespace stlib
{
namespace numerical {

//! Whether to use branching and meta-programming.
template<bool _IsDynamic, bool _UseBranching, typename _Integer>
class TraitsForDynamicAndBranchingAndInteger {
public:
   //! Whether it is dynamic.
   BOOST_STATIC_CONSTEXPR bool IsDynamic = _IsDynamic;
   //! Whether to use branching.
   BOOST_STATIC_CONSTEXPR bool UseBranching = _UseBranching;
   //! The integer type.
   typedef _Integer Integer;
};


//! Probability mass function for a discrete generator.
/*!
  Manage the probability mass function.
*/
template < class PmfAndSum = DgPmfAndSum<false>,
         class Traits = TraitsForDynamicAndBranchingAndInteger < false /*CONTINUE*/, true, unsigned > >
class DgPmfInteger;


//! Probability mass function for a discrete generator.
/*!
  Manage the probability mass function.  Use an array of unsigned integers
  that represent the probabilities with fixed precision.

  \note  This is just a test.  It does not yet have dynamic capability.
*/
template<class PmfAndSum, bool _UseBranching, typename _Integer>
class DgPmfInteger<PmfAndSum, TraitsForDynamicAndBranchingAndInteger<false, _UseBranching, _Integer> > :
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

   //
   // More private types.
   //
private:

   //! The traits.
   typedef TraitsForDynamicAndBranchingAndInteger<false, _UseBranching, _Integer> Traits;
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
   DgPmfInteger() :
      Base(),
      // The PMF array is empty.
      _ipmf() {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DgPmfInteger(ForwardIterator begin, ForwardIterator end) :
      Base(),
      _ipmf() {
      initialize(begin, end);
   }

   //! Copy constructor.
   DgPmfInteger(const DgPmfInteger& other) :
      Base(other),
      _ipmf(other._ipmf) {}

   //! Assignment operator.
   DgPmfInteger&
   operator=(const DgPmfInteger& other) {
      if (this != &other) {
         Base::operator=(other);
         _ipmf = other._ipmf;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfInteger() {}

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
     return computeDeviate(r, offset, std::integral_constant<bool,
                           Traits::UseBranching>());
   }

   //! Return a discrete deviate.
   /*!
     Use a linear search to sum probabilities until the sum reaches r.
   */
   std::size_t
   operator()(const unsigned r) const {
     return computeDeviate(r, std::integral_constant<bool,
                           Traits::UseBranching>());
   }

private:

   std::size_t
   computeDeviate(const unsigned r, const std::size_t offset,
                  std::true_type /*UseBranching*/) const {
      return offset +
             linearSearchChopDownUnguarded(_ipmf.begin() + offset, _ipmf.end(), r);
   }

   std::size_t
   computeDeviate(const unsigned r, const std::size_t offset,
                  std::false_type /*UseBranching*/) const {
      // CONTINUE:  I divide by 2 to get an integer.
      return offset +
             linearSearchChopDownNoBranching(_ipmf.begin() + offset, _ipmf.end(),
                                             int(r / 2));
   }

   std::size_t
   computeDeviate(const unsigned r, std::true_type /*UseBranching*/)
   const {
      return linearSearchChopDownUnguarded(_ipmf.begin(), _ipmf.end(), r);
   }

   std::size_t
   computeDeviate(const unsigned r, std::false_type /*UseBranching*/)
   const {
      // CONTINUE:  I divide by 2 to get an integer.
      return linearSearchChopDownNoBranching(_ipmf.begin(), _ipmf.end(),
                                             int(r / 2));
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   using Base::operator[];
   //! Get the beginning of the PMF.
   using Base::begin;
   //! Get the end of the PMF.
   using Base::end;
   //! Get the number of possible deviates.
   using Base::size;
   //! Get the sum of the probability mass functions.
   using Base::sum;
   //! Return true if the sum of the PMF is positive.
   using Base::isValid;

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgPmfInteger& other) const {
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
   initialize(ForwardIterator start, ForwardIterator finish) {
      // Initialize the floating point PMF and the sum.
      Base::initialize(start, finish);

      //
      // Initialize the fixed precision PMF.
      // The integer array will sum to exactly
      // std::numeric_limits<Integer>::max().
      //

      // Resize the array.
      _ipmf.resize(size());
      Number s = sum();
      Integer iSum = std::numeric_limits<Integer>::max();
      for (std::size_t i = 0; i < _ipmf.size() - 1; ++i) {
         // Note: I use getPmfBeginning() instead of getPmf() because the
         // array may be sorted.
         _ipmf[i] = Integer(Number(iSum) * begin()[i] / s);
         s -= begin()[i];
         iSum -= _ipmf[i];
      }
      // Set the last element.
      *(_ipmf.end() - 1) = iSum;
   }

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

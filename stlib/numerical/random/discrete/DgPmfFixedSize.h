// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfFixedSize.h
  \brief Probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmfFixedSize_h__)
#define __numerical_DgPmfFixedSize_h__

#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/ext/vector.h"

#include <limits>

namespace stlib
{
namespace numerical {

//! Whether to use branching and meta-programming.
template<bool Branching, bool MetaProgramming>
class TraitsForBranchingAndMetaProgramming {
public:
   //! Whether to use branching.
   BOOST_STATIC_CONSTEXPR bool UseBranching = Branching;
   //! Whether to use meta-programming.
   BOOST_STATIC_CONSTEXPR bool UseMetaProgramming = MetaProgramming;
};


//! Probability mass function for a discrete generator.
/*!
  \param N The number of elements.

  Manage the probability mass function.
*/
template < std::size_t N,
         class Traits = TraitsForBranchingAndMetaProgramming<false, false> >
class DgPmfFixedSize {
   //
   // Public types.
   //
public:

   //! The number type.
   typedef double Number;

   //
   // Private types.
   //
private:

   //! The array type.
   /*!
     It is more efficient to use a dynamically sized array than one of
     fixed size. CONTINUE: I think this statement applies to FixedArray.
     Is the same true of std::array?
   */
   typedef std::vector<Number> Container;

   //
   // More public types.
   //
public:

   //! An iterator on the probabilities.
   typedef typename Container::iterator Iterator;
   //! A const iterator on the probabilities.
   typedef typename Container::const_iterator ConstIterator;

   //
   // Member data.
   //
private:

   //! Probability mass function.  (This is scaled and may not sum to unity.)
   Container _pmf;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfFixedSize() :
      // Allocate memory for the array.
      _pmf(N) {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DgPmfFixedSize(ForwardIterator begin, ForwardIterator end) :
      // Allocate memory for the array.
      _pmf(N) {
      initialize(begin, end);
   }

   //! Copy constructor.
   DgPmfFixedSize(const DgPmfFixedSize& other) :
      _pmf(other._pmf) {}

   //! Assignment operator.
   DgPmfFixedSize&
   operator=(const DgPmfFixedSize& other) {
      if (this != &other) {
         _pmf = other._pmf;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfFixedSize() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
protected:

   //! Return a discrete deviate.
   /*!
     Use a linear search to sum probabilities until the sum reaches r.
   */
   std::size_t
   operator()(Number r) const {
     return computeDeviate(r, std::integral_constant<bool,
                           Traits::UseBranching>(),
                           std::integral_constant<bool,
                           Traits::UseMetaProgramming>());
   }

private:

   std::size_t
   computeDeviate(Number r, std::false_type /*UseBranching*/,
                  std::false_type /*UseMetaProgramming*/) const {
      std::size_t result = 0;
      for (std::size_t i = 0 ; i != N - 1; ++i) {
         r -= _pmf[i];
         result += (r >= 0);
      }
      return result;
   }

   std::size_t
   computeDeviate(Number r, std::true_type /*UseBranching*/,
                  std::false_type /*UseMetaProgramming*/) const {
      std::size_t i = 0;
      for (; i != N - 1; ++i) {
         if (r < _pmf[i]) {
            break;
         }
         r -= _pmf[i];
      }
      return i;

#if 0
      std::size_t i = 0;
      for (; i != N - 1; ++i) {
         r -= _pmf[i];
         if (r < 0) {
            break;
         }
      }
      return i;
#endif
   }

   std::size_t
   computeDeviate(Number r, std::false_type /*UseBranching*/,
                  std::true_type /*UseMetaProgramming*/) const {
      return LinearSearchChopDown<N, ConstIterator>::result(_pmf.begin(), r);
   }


#if 0
   return (r >= _pmf[0]) + (r >= _pmf[0] + _pmf[1]) +
          (r >= _pmf[0] + _pmf[1] + _pmf[2]);
#endif

#if 0
   r -= _pmf[0];
   std::size_t i = (r >= 0);
   r -= _pmf[1];
   i += (r >= 0);
   r -= _pmf[2];
   i += (r >= 0);
   return i;
#endif

#if 0
   std::size_t i = (r >= _pmf[0]);
   r -= _pmf[0];
   i += (r >= _pmf[1]);
   r -= _pmf[1];
   i += (r >= _pmf[2]);
   return i;
#endif

#if 0
   if (r < _pmf[0]) {
      return 0;
   }
   r -= _pmf[0];

   if (r < _pmf[1]) {
      return 1;
   }
   r -= _pmf[1];

   if (r < _pmf[2]) {
      return 2;
   }

   return 3;
#endif

#if 0
   r -= _pmf[0];
   if (r < 0) {
      return 0;
   }

   r -= _pmf[1];
   if (r < 0) {
      return 1;
   }

   r -= _pmf[2];
   if (r < 0) {
      return 2;
   }

   return 3;
#endif



   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   Number
   getPmf(const std::size_t index) const {
      return _pmf[index];
   }

   //! Get the number of possible deviates.
   std::size_t
   getSize() const {
      return N;
   }

protected:

   //! Get the beginning of the probabilities in the PMF.
   ConstIterator
   getPmfBeginning() const {
      return _pmf.begin();
   }

   //! Get the end of the probabilities in the PMF.
   ConstIterator
   getPmfEnd() const {
      return _pmf.end();
   }

   //! Get the index of the specified element.
   std::size_t
   getIndex(const std::size_t n) const {
      return n;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgPmfFixedSize& other) const {
      return _pmf == other._pmf;
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
      assert(std::distance(begin, end) == std::ptrdiff_t(getSize()));
      std::copy(begin, end, _pmf.begin());
   }

   //! Set the probability mass function with the specified index.
   void
   setPmf(std::size_t index, Number value) {
      _pmf[index] = value;
   }

   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   setPmf(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         _pmf[i] = iterator[i];
      }
   }

protected:

   //! Do nothing.
   void
   updatePmf() {
   }

   //! Get the beginning of the probabilities in the PMF.
   Iterator
   getPmfBeginning() {
      return _pmf.begin();
   }

   //! Get the end of the probabilities in the PMF.
   Iterator
   getPmfEnd() {
      return _pmf.end();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "PMF = \n" << _pmf << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif

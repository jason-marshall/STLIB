// -*- C++ -*-

/*!
  \file numerical/random/discrete/linearSearch.h
  \brief Probability mass function for a discrete generator.
*/

#if !defined(__numerical_random_discrete_linearSearch_h__)
#define __numerical_random_discrete_linearSearch_h__

// CONTINUE
#include <iostream>

#include <iterator>
#include <limits>

namespace stlib
{
namespace numerical {

//! Linear search with the chop-down method.  Do not use branching.
template<typename InputConstIterator, typename T>
std::size_t
linearSearchChopDownNoBranching(InputConstIterator begin,
                                InputConstIterator end, T r);

//! Unguarded linear search with the chop-down method.
template<typename RandomAccessConstIterator, typename T>
std::size_t
linearSearchChopDownUnguarded(RandomAccessConstIterator begin,
                              RandomAccessConstIterator end, T r);


//! Guarded linear search with the chop-down method.
template<typename RandomAccessConstIterator, typename T>
std::size_t
linearSearchChopDownGuarded(RandomAccessConstIterator begin,
                            RandomAccessConstIterator end, T r);

//! Guarded linear search on a range of value/index pairs with the chop-down method.
template<typename RandomAccessConstIterator, typename T>
std::size_t
linearSearchChopDownGuardedPair(RandomAccessConstIterator begin,
                                RandomAccessConstIterator end, T r);

// CONTINUE REMOVE
//! Linear search with the chop-down method.
template<std::size_t N, typename Iterator>
class LinearSearchChopDown {
public:
   //! The number type.
   typedef typename std::iterator_traits<Iterator>::value_type Number;

   //! Linear search without branching.
   static
   Number
   result(Iterator i, Number r, std::false_type /*UseBranching*/) {
      r -= *i;
      return (r >= 0) + LinearSearchChopDown < N - 1, Iterator >::result
        (i + 1, r, std::false_type());
#if 0
      return (r >= *i) + LinearSearchChopDown < N - 1, Iterator >::result
        (i + 1, r - *i, std::false_type());
#endif
   }

   //! Linear search with branching.
   static
   Number
   result(Iterator i, Number r, std::true_type /*UseBranching*/) {
      if (r < *i) {
         return 0;
      }
      return 1 + LinearSearchChopDown < N - 1, Iterator >::result
        (i + 1, r - *i, std::true_type());
   }

#if 0
   // Build-up version.
   static
   Number
   result(Iterator i, Number r, Number s) {
      return (r >= s + *i) +
             LinearSearchChopDown < N - 1, Iterator >::result(i + 1, r, s + *i);
   }
#endif
};

//! Linear search. Specialization for one element.
template<typename Iterator>
class LinearSearchChopDown<1, Iterator> {
public:
   //! The number type.
   typedef typename std::iterator_traits<Iterator>::value_type Number;

   //! Perform the linear search.
   template<bool UseBranching>
   static
   Number
   result(Iterator, Number,
          std::integral_constant<bool, UseBranching> /*dummy*/) {
      return 0;
   }

#if 0
   // Build-up version.
   static
   Number
   result(Iterator i, Number r, Number s) {
      return 0;
   }
#endif
};


} // namespace numerical
}

#define __numerical_random_discrete_linearSearch_ipp__
#include "stlib/numerical/random/discrete/linearSearch.ipp"
#undef __numerical_random_discrete_linearSearch_ipp__

#endif

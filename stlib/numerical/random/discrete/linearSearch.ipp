// -*- C++ -*-

#if !defined(__numerical_random_discrete_linearSearch_ipp__)
#error This file is an implementation detail of linearSearch.
#endif

namespace stlib
{
namespace numerical {

template<typename InputConstIterator, typename T>
inline
std::size_t
linearSearchChopDownNoBranching(InputConstIterator begin,
                                const InputConstIterator end, T r) {
   static_assert(std::numeric_limits<T>::is_signed,
                 "The number type must be signed.");
   std::size_t result = 0;
   while (begin != end - 1) {
      r -= *begin;
      ++begin;
      result += (r > 0);
   }
   return result;
}


// Version for signed number types.
template<typename RandomAccessConstIterator, typename T>
inline
std::size_t
linearSearchChopDownUnguarded(RandomAccessConstIterator begin,
                              RandomAccessConstIterator end, T r,
                              std::true_type /*isSigned*/) {
   // Step until the accumulated probability is greater than or equal to the
   // random variable.
   RandomAccessConstIterator i = begin;
   for (; i != end && (r -= *i) > 0; ++i) {
   }
   // Return the index of the PMF.  Handle the special case that round-off errors
   // make us reach the end.  (Using an if statement to do this would be a
   // little more expensive.)
   return i - begin - (i == end);

   // This method checks the validity of the result.
#if 0
   // If we made it all the way to the end due to round-off error.
   if (i == end && r >= 0) {
      // Step back to the last valid (non-zero) element.
      --i;
      while (i != begin && *i == 0) {
         --i;
      }
#ifdef STLIB_DEBUG
      // There must be non-zero probabilities.
      assert(*i != 0);
#endif
      // Return the index of the PMF.
      return i - begin;
   }
   // Step back to the probability that exceeded r.
   // Return the index of the PMF.
   return i - begin - 1;
#endif
}


// Version for unsigned number types.
template<typename RandomAccessConstIterator, typename T>
inline
std::size_t
linearSearchChopDownUnguarded(RandomAccessConstIterator begin,
                              RandomAccessConstIterator end, T r,
                              std::false_type /*isSigned*/) {
   // Step until the accumulated probability is greater than or equal to the
   // random variable.
   RandomAccessConstIterator i = begin;
   for (; i != end && r > *i; r -= *i, ++i) {
   }
   // Return the index of the PMF.  Handle the special case that round-off errors
   // make us reach the end.  (Using an if statement to do this would be a
   // little more expensive.)
   return i - begin - (i == end);
}


template<typename RandomAccessConstIterator, typename T>
inline
std::size_t
linearSearchChopDownUnguarded(RandomAccessConstIterator begin,
                              RandomAccessConstIterator end, T r) {
   // Call the signed or unsigned version.
   return linearSearchChopDownUnguarded
     (begin, end, r, std::integral_constant<bool,
      std::numeric_limits<T>::is_signed>());
}


template<typename RandomAccessConstIterator, typename T>
inline
std::size_t
linearSearchChopDownGuarded(RandomAccessConstIterator begin,
                            RandomAccessConstIterator end, T r) {
   // Step until the accumulated probability is greater than or equal to the
   // random variable.
   RandomAccessConstIterator i = begin;
   for (; (r -= *i) > 0; ++i) {
   }
   // Return the index of the PMF.  Handle the special case that round-off errors
   // make us reach the end.  (Using an if statement to do this would be a
   // little more expensive.)
   return i - begin - (i == end);


   // REMOVE
#if 0
   BidirectionalConstIterator i = begin;
   // The >= is important.  Otherwise, we could return a zero probability.
   while (r > 0) {
      r -= *i++;
   }

   // If we made it all the way to the guard element due to round-off error.
   if (i == end) {
      // Step back to the last valid (non-zero) element (skip the guard
      // element).
      // Return the index of the PMF.
      return i - begin - 2;
   }
   // Step back to the probability that exceeded r.
   // Return the index of the PMF.
   return i - begin - 1;
#endif

   // This method checks the validity of the result.
#if 0
   // If we made it all the way to the guard element due to round-off error.
   if (i == end) {
      // Step back to the last valid (non-zero) element (skip the guard
      // element).
      i -= 2;
      while (i != begin && *i == 0) {
         --i;
      }
#ifdef STLIB_DEBUG
      // There must be non-zero probabilities.
      assert(*i != 0);
#endif
   }
   else {
      // Step back to the probability that exceeded r.
      --i;
   }
   // Return the index of the PMF.
   return i - begin;
#endif
}

template<typename RandomAccessConstIterator, typename T>
inline
std::size_t
linearSearchChopDownGuardedPair(RandomAccessConstIterator begin,
                                RandomAccessConstIterator end, T r) {
   // Step until the accumulated probability is greater than or equal to the
   // random variable.
   RandomAccessConstIterator i = begin;
   for (; (r -= i->first) > 0; ++i) {
   }
   // Return the index of the PMF.  Handle the special case that round-off errors
   // make us reach the end.  (Using an if statement to do this would be a
   // little more expensive.)
   return i->second - (i == end);
}

} // namespace numerical
}

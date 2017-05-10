// -*- C++ -*-

#if !defined(__numerical_random_DiscreteGeneratorBinarySearchRecursiveCdf_ipp__)
#error This file is an implementation detail of DiscreteGeneratorBinarySearchRecursiveCdf.
#endif

namespace stlib
{
namespace numerical {

// Return a discrete deviate.
template<class Generator>
inline
typename DiscreteGeneratorBinarySearchRecursiveCdf<Generator>::result_type
DiscreteGeneratorBinarySearchRecursiveCdf<Generator>::
operator()() {
   Number r = _continuousUniformGenerator() * sum();
   Number x;
   std::size_t index = -1;
   std::size_t step = _partialRecursiveCdf.size() / 2;
   // The for loop is slightly faster.
   for (std::size_t i = _indexBits; i != 0; --i) {
      //while (step != 0) {
      x = _partialRecursiveCdf[index + step];

      // It is faster whithout the branch.
#if 0
      if (x <= r) {
         r -= x;
         index += step;
      }
#endif
      const bool shouldMove = x <= r;
      r -= shouldMove * x;
      index += shouldMove * step;

      step /= 2;
   }
   // This is the weighted probability that exceeds r.
   ++index;

   // Step back over zero probabilities if necessary.
   while (_pmf[index] == 0) {
      --index;
   }

   return index;
}


// Update the data structure following calls to setPmf() .
template<class Generator>
inline
void
DiscreteGeneratorBinarySearchRecursiveCdf<Generator>::
repair() {
   // Start with the (zero-padded) PMF.
   std::copy(_pmf.begin(), _pmf.end(), _partialRecursiveCdf.begin());
   std::fill(_partialRecursiveCdf.begin() + size(),
             _partialRecursiveCdf.end(), Number(0));

   // Compute the partial, recursive sums.
   for (std::size_t step = 2, offset = 1; step <= _partialRecursiveCdf.size();
         step *= 2, offset *= 2) {
      for (std::size_t i = step - 1, j = step - 1 - offset;
            i < _partialRecursiveCdf.size(); i += step, j += step) {
         _partialRecursiveCdf[i] += _partialRecursiveCdf[j];
      }
   }

   // The initial error in the PMF sum.
   _error = size() * sum() * std::numeric_limits<Number>::epsilon();
}


//! Update the CDF.
template<class Generator>
inline
void
DiscreteGeneratorBinarySearchRecursiveCdf<Generator>::
updateCdf(const std::size_t index, const Number difference) {
   // Shift back by one so we can use offsets.
   typename std::vector<Number>::iterator array
   = _partialRecursiveCdf.begin() - 1;

   std::size_t offset = _partialRecursiveCdf.size();
   for (std::ptrdiff_t shift = _indexBits; shift >= 0; --shift, offset /= 2) {
      //for (; offset != 0; offset /= 2) {
      // Check the appropriate bit.
      if (index & offset) {
         array += offset;
      }
      else {
         array[offset] += difference;
      }
#if 0
      // Little performance difference.
      const bool bit = index & offset;
      array[offset] += (! bit) * difference;
      array += bit * offset;
#endif
   }

   // Not as fast.
#if 0
   std::size_t offset = _partialRecursiveCdf.size();
   for (std::ptrdiff_t shift = _indexBits; shift >= 0; --shift, offset /= 2) {
      // Extract the shift_th digit.
      if ((index >> shift) & 1) {
         array += offset;
      }
      else {
         array[offset] += difference;
      }
      // This is not as fast.
#if 0
      const bool bit = (index >> shift) & 1;
      array[offset] += (! bit) * difference;
      array += bit * offset;
#endif
   }
#endif
}

} // namespace numerical
}

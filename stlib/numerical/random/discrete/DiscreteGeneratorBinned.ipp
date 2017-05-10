// -*- C++ -*-

#if !defined(__numerical_random_DiscreteGeneratorBinned_ipp__)
#error This file is an implementation detail of DiscreteGeneratorBinned.
#endif

namespace stlib
{
namespace numerical {

template<class Generator>
inline
typename DiscreteGeneratorBinned<Generator>::result_type
DiscreteGeneratorBinned<Generator>::
operator()() {
   const Number efficiency = computeEfficiency();
   // If the efficiency is very low (it has fallen below the minimum allow
   // efficiency) or if the efficiency is low (below the target efficiency)
   // and it is time for a rebuild.
   if (efficiency < getMinimumEfficiency() ||
         (efficiency < getTargetEfficiency() && _stepsUntilNextRebuild <= 0)) {
      rebuild();
   }
   // If it is time for a repair.
   else if (_stepsUntilNextRepair <= 0) {
      repair();
   }

   // Loop until the point is not rejected.
   for (;;) {
      unsigned random = (*_discreteUniformGenerator)();
      // Use the first bits for indexing.
      unsigned index = random & IndexMask();
      // Use the remaining bits for the height deviate.
      unsigned heightGenerator = random >> IndexBits();
      Number height = heightGenerator * _heightUpperBound * MaxHeightInverse();
      // If we have a hit for the PMF's in this bin.
      if (height < _binnedPmf[index]) {
         // Do a linear search to find the PMF that we hit.
         std::size_t pmfIndex = _deviateIndices[index];
         for (; pmfIndex < _deviateIndices[index + 1] - 1; ++pmfIndex) {
            height -= _pmf[pmfIndex];
            if (height <= 0) {
               break;
            }
         }
         return _permutation[pmfIndex];
      }
   }
}


// Initialize the probability mass function.
template<class Generator>
template<typename ForwardIterator>
inline
void
DiscreteGeneratorBinned<Generator>::
initialize(ForwardIterator begin, ForwardIterator end) {
   // Build the array for the PMF.
   _pmf.resize(std::distance(begin, end));
   std::copy(begin, end, _pmf.begin());
   _permutation.resize(_pmf.size());
   _rank.resize(_pmf.size());
   _binIndices.resize(_pmf.size() + 1);

   // Initialize so the efficiency appears to be zero.
   _pmfSum = 0;
   _heightUpperBound = 1;
   // Rebuild the data structure by sorting the PMF and packing them into bins.
   rebuild();
}


// Rebuild the bins.
template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
rebuild() {
   _stepsUntilNextRebuild = _stepsBetweenRebuilds;
   // Rebuilding also repairs the data structure, so we reset that counter
   // as well.
   _stepsUntilNextRepair = _stepsBetweenRepairs;

   //
   // Sort the PMF array in descending order.
   //
   _pmfSum = ext::sum(_pmf);
   for (std::size_t i = 0; i != _permutation.size(); ++i) {
      _permutation[i] = i;
   }
   // Sort in descending order.
   ads::sortTogether(_pmf.begin(), _pmf.end(), _permutation.begin(),
                     _permutation.end(), std::greater<Number>());
   // Compute the ranks.
   for (std::size_t i = 0; i != _permutation.size(); ++i) {
      _rank[_permutation[i]] = i;
   }

   packIntoBins();
   _heightUpperBound = *std::max_element(_binnedPmf.begin(), _binnedPmf.end());
}



// Set the probability mass function with the specified index.
template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
setPmf(const std::size_t index, const Number value) {
   // The index in the re-ordered PMF.
   const std::size_t i = _rank[index];

   // If the value has not changed, do nothing.  I need this check; otherwise
   // the following branch could be expensive.
   if (_pmf[i] == value) {
      return;
   }

   // If the PMF has become zero. (It was not zero before.)
   if (value == 0) {
      // Set the PMF to zero.
      _pmf[i] = 0;
      // Repair the data structure.  This is necessary to ensure that the
      // binned PMF are correct.  They must be exactly zero.  Likewize, the
      // sum of the PMF may have become zero.
      repair();
      return;
   }

   // The remainder of this function is the standard case.  Update the data
   // structure using the difference between the new and old values.

   const Number difference = value - _pmf[i];
   // Update the sum of the PMF.
   _pmfSum += difference;
   // Update the PMF array.
   _pmf[i] = value;

   //
   // Update the binned PMF.
   //
   const std::size_t binIndex = _binIndices[i];
   _binnedPmf[binIndex] += difference;

   // Update the upper bound on the bin height.
   if (_binnedPmf[binIndex] > _heightUpperBound) {
      _heightUpperBound = _binnedPmf[binIndex];
   }

   // Fix the bin if necessary.
   if (i < _splittingEnd && _binnedPmf[binIndex] < 0) {
      fixBin(binIndex);
   }

   --_stepsUntilNextRepair;
   --_stepsUntilNextRebuild;
}


// Update the data structure following calls to setPmfWithoutUpdating() .
template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
updatePmf() {
   //
   // Compute the binned PMF.
   //
   std::fill(_binnedPmf.begin(), _binnedPmf.end(), 0);
   // First do the PMF's that are split over multiple bins.
   for (std::size_t i = 0; i != _splittingEnd; ++i) {
      // Split the PMF over a number of bins.
      const Number height = _pmf[i] / (_binIndices[i + 1] - _binIndices[i]);
      for (std::size_t j = _binIndices[i]; j != _binIndices[i + 1]; ++j) {
         _binnedPmf[j] = height;
      }
   }
   // Then do the PMF's that sit in a single bin.
   for (std::size_t i = _splittingEnd; i != _pmf.size(); ++i) {
      _binnedPmf[_binIndices[i]] += _pmf[i];
   }

   // Compute the sum of the PMF.
   // Choose the more efficient method.
   if (_pmf.size() < _binnedPmf.size()) {
      _pmfSum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);
   }
   else {
      // Sum over the bins.
      _pmfSum = std::accumulate(_binnedPmf.begin(), _binnedPmf.end(), 0.0);
   }

   // Compute the upper bound on the bin height.
   _heightUpperBound = *std::max_element(_binnedPmf.begin(), _binnedPmf.end());
}


// Pack the block into bins.
template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
packIntoBins() {
   const Number height = computeBinHeight<Number>(_pmf.begin(), _pmf.end(),
                         NumberOfBins);
   const Number inverse = 1.0 / height;

   // Empty the bins.
   std::fill(_binnedPmf.begin(), _binnedPmf.end(), 0);
   std::fill(_deviateIndices.begin(), _deviateIndices.end(), _pmf.size());

   // Pack the blocks that are split across multiple bins.
   std::size_t pmfIndex = 0, binIndex = 0;
   for (; pmfIndex != _pmf.size() && _pmf[pmfIndex] > height; ++pmfIndex) {
      _binIndices[pmfIndex] = binIndex;
      const std::size_t count = std::size_t(_pmf[pmfIndex] * inverse) + 1;
      const Number binHeight = _pmf[pmfIndex] / count;
      for (std::size_t i = 0; i != count; ++i) {
         _deviateIndices[binIndex] = pmfIndex;
         _binnedPmf[binIndex] = binHeight;
         ++binIndex;
      }
   }
   // Record the end of the PMF's that are split accross multiple bins.
   _splittingEnd = pmfIndex;

   //
   // Pack the stacking bins.
   //

   // If there are blocks left to stack.
   if (pmfIndex != _pmf.size()) {
      // Put a block in the current bin.
      _binIndices[pmfIndex] = binIndex;
      _deviateIndices[binIndex] = pmfIndex;
      _binnedPmf[binIndex] = _pmf[pmfIndex];
      ++pmfIndex;

      // Pack the rest of the blocks.
      Number newHeight;
      while (pmfIndex != _pmf.size()) {
         // Try adding a block to the current bin.
         newHeight = _binnedPmf[binIndex] + _pmf[pmfIndex];
         // If we can fit it in the current bin.
         if (newHeight <= height) {
            // Add the block to the bin.
            _binIndices[pmfIndex] = binIndex;
            _binnedPmf[binIndex] = newHeight;
            // Move to the next block.
            ++pmfIndex;
         }
         else {
            // Put the block in the next bin.
            ++binIndex;
            _binIndices[pmfIndex] = binIndex;
            _deviateIndices[binIndex] = pmfIndex;
            _binnedPmf[binIndex] = _pmf[pmfIndex];
            ++pmfIndex;
         }
      }

      // Move to an empty bin.
      ++binIndex;
   }
   // The guard value.
   _binIndices[pmfIndex] = binIndex;
}


template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
fixBin(const std::size_t binIndex) {
   for (std::size_t j = binIndex + 1;
         _deviateIndices[binIndex] == _deviateIndices[j] &&
         _binnedPmf[binIndex] < 0; ++j) {
      _binnedPmf[binIndex] += _binnedPmf[j];
      _binnedPmf[j] = 0;
   }
}


// Print information about the data structure.
template<class Generator>
inline
void
DiscreteGeneratorBinned<Generator>::
print(std::ostream& out) const {
   out << "Bin data:\n\n"
       << "Height upper bound = " << _heightUpperBound << "\n"
       << "Binned PMF = " << _binnedPmf << "\n"
       << "Deviate indices = " << _deviateIndices << "\n"
       << "\nPMF data:\n\n"
       << "PMF sum = " << _pmfSum << "\n"
       << "Splitting end = " << _splittingEnd << "\n"
       << "PMF = \n" << _pmf << '\n'
       << "Permutation = \n" << _permutation << '\n'
       << "Rank = \n" << _rank << '\n'
       << "Bin indices = " << _binIndices << '\n'
       << "Steps between repairs = " << _stepsBetweenRepairs << "\n"
       << "Steps until next repair = " << _stepsUntilNextRepair << "\n"
       << "Steps between rebuilds = " << _stepsBetweenRebuilds << "\n"
       << "Steps until next rebuild = " << _stepsUntilNextRebuild << "\n"
       << "Target efficiency = " << _targetEfficiency << "\n"
       << "Minimum efficiency = " << _minimumEfficiency << "\n";
}

} // namespace numerical
}

// -*- C++ -*-

#if !defined(__numerical_random_DiscreteGeneratorBinsSplitting_ipp__)
#error This file is an implementation detail of DiscreteGeneratorBinsSplitting.
#endif

namespace stlib
{
namespace numerical {

//----------------------------------------------------------------------------
// Static class.
//----------------------------------------------------------------------------

template<class Generator>
inline
typename DiscreteGeneratorBinsSplitting<false, Generator>::result_type
DiscreteGeneratorBinsSplitting<false, Generator>::
operator()() {
   // Loop until the point is not rejected.
   for (;;) {
      unsigned random = (*_discreteUniformGenerator)();
      // Use the first bits for indexing.
      unsigned index = random & getIndexMask();
      // Use the remaining bits for the height deviate.
      unsigned heightGenerator = random >> getIndexBits();
      Number height = heightGenerator * _heightUpperBound * getMaxHeightInverse();
      // If we have a hit for the PMF in this bin.
      if (height < _binnedPmf[index]) {
         return _deviateIndices[index];
      }
   }
}

// Initialize the probability mass function.
template<class Generator>
template<typename ForwardIterator>
inline
void
DiscreteGeneratorBinsSplitting<false, Generator>::
initialize(ForwardIterator begin, ForwardIterator end) {
   // Build the array for the PMF.
   _pmf.resize(std::distance(begin, end));
   std::copy(begin, end, _pmf.begin());
   _binIndices.resize(_pmf.size() + 1);

   // Make sure that there are no more probabilities than bins.
   assert(_pmf.size() <= _binnedPmf.size());

   // Initialize so the efficiency appears to be zero.
   _pmfSum = 0;
   _heightUpperBound = 1;
   // Rebuild the data structure by splitting the PMF across the bins.
   rebuild();
}


// Rebuild the bins.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<false, Generator>::
rebuild() {
  _pmfSum = ext::sum(_pmf);
  packIntoBins();
  _heightUpperBound = *std::max_element(_binnedPmf.begin(), _binnedPmf.end());
}


// Pack the blocks into bins.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<false, Generator>::
packIntoBins() {
   // The inverse of the maximum bin height.
   const Number inverseMaxBinHeight = (getNumberOfBins() - _pmf.size() + 1) /
                                      (4.001 * _pmfSum);

   //
   // Generate an initial guess for the bin counts.
   //

   // Temporarily use the bin indices array to record these.
   for (std::size_t i = 0; i != _pmf.size(); ++i) {
      _binIndices[i] = std::max(std::size_t(1),
                                std::size_t(_pmf[i] * inverseMaxBinHeight));
   }
   *(_binIndices.end() - 1) = 0;

   // Temporarily use the binned PMF array to record the split heights.
   for (std::size_t i = 0; i != _pmf.size(); ++i) {
      _binnedPmf[i] = _pmf[i] / _binIndices[i];
   }

   // CONTINUE: This could be implemented more efficiently with a heap.
   // (At least it would be more efficient for large problems.)
   std::size_t numberOfUsedBins = ext::sum(_binIndices);
   while (numberOfUsedBins++ != _binnedPmf.size()) {
      // Find the PMF that we should expand into one more bin.
      const std::size_t i = std::max_element(_binnedPmf.begin(),
                                             _binnedPmf.begin() + _pmf.size())
                            - _binnedPmf.begin();
      // Expand it into one more bin.
      ++_binIndices[i];
      _binnedPmf[i] = _pmf[i] / _binIndices[i];
   }

   // Set the binned PMF and the deviate indices.
   std::size_t index = 0;
   for (std::size_t i = 0; i != _pmf.size(); ++i) {
      const Number binHeight = _pmf[i] / _binIndices[i];
      for (std::size_t j = 0; j != _binIndices[i]; ++j) {
         _binnedPmf[index] = binHeight;
         _deviateIndices[index] = i;
         ++index;
      }
   }
   assert(index == _binnedPmf.size());
   _deviateIndices[index] = _pmf.size();

   // Convert the bin counts to bin indices.
   for (std::size_t i = _binIndices.size() - 1; i > 0; --i) {
      _binIndices[i] = _binIndices[i-1];
   }
   _binIndices[0] = 0;
   std::partial_sum(_binIndices.begin(), _binIndices.end(),
                    _binIndices.begin());
   assert(_binIndices.back() == getNumberOfBins());
}


template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<false, Generator>::
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
DiscreteGeneratorBinsSplitting<false, Generator>::
print(std::ostream& out) const {
   out << "Bin data:\n\n"
       << "Height upper bound = " << _heightUpperBound << "\n"
       << "Binned PMF = " << _binnedPmf << '\n'
       << "Deviate indices = " << _deviateIndices << '\n'
       << "\nPMF data:\n\n"
       << "PMF sum = " << _pmfSum << "\n"
       << "PMF = \n" << _pmf << '\n'
       << "Bin indices = " << _binIndices << '\n';
}


//----------------------------------------------------------------------------
// Dynamic class.
//----------------------------------------------------------------------------

// Initialize the probability mass function.
template<class Generator>
template<typename ForwardIterator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
initialize(ForwardIterator begin, ForwardIterator end) {
   RepairBase::resetRepairCounter();
   Base::initialize(begin, end);

#ifdef MODIFY
   // For each probability, start at the first bin.
   _binsToModify.resize(getSize());
   std::copy(_binIndices.begin(), _binIndices.end() - 1, _binsToModify.begin());
#endif

   updateMinimumAllowedEfficiency();
}


// Rebuild the bins.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
rebuild() {
#if 0
   static std::size_t count = 1;
   std::cout << "rebuild " << count++ << "\n";
#endif
   Base::rebuild();

#ifdef MODIFY
   // For each probability, start at the first bin.
   std::copy(_binIndices.begin(), _binIndices.end() - 1, _binsToModify.begin());
#endif

   updateMinimumAllowedEfficiency();
   // Rebuilding also repairs the data structure, so we reset that counter.
   resetRepairCounter();
}



// Set the probability mass function with the specified index.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
set(const std::size_t index, const Number value) {
   // If the value has not changed, do nothing.  I need this check; otherwise
   // the following branch could be expensive.
   if (_pmf[index] == value) {
      return;
   }

   // If the PMF has become zero. (It was not zero before.)
   if (value == 0) {
      // Set the PMF to zero.
      _pmf[index] = 0;
      // Repair the data structure.  This is necessary to ensure that the
      // binned PMF are correct.  They must be exactly zero.  Likewise, the
      // sum of the PMF may have become zero.
      repair();
      return;
   }

   // The remainder of this function is the standard case.  Update the data
   // structure using the difference between the new and old values.

   const Number difference = value - _pmf[index];
   // Update the sum of the PMF.
   _pmfSum += difference;
   // Update the PMF array.
   _pmf[index] = value;

   //
   // Update the binned PMF.
   //
#ifdef MODIFY
   // Get the bin to modify.
   const std::size_t binIndex = _binsToModify[index];
   // We'll use the following bin next time.
   _binsToModify[index] = _binIndices[index] +
                          (_binsToModify[index] - _binIndices[index] + 1) %
                          (_binIndices[index + 1] - _binIndices[index]);
#else
   // Get the bin to modify.
   const std::size_t binIndex = _binIndices[index];
#endif
   _binnedPmf[binIndex] += difference;

   // Update the upper bound on the bin height.
   if (_binnedPmf[binIndex] > _heightUpperBound) {
      _heightUpperBound = _binnedPmf[binIndex];
   }

   // Fix the bin if necessary.
   if (_binnedPmf[binIndex] < 0) {
#ifdef MODIFY
      repair();
#else
      Base::fixBin(binIndex);
#endif
   }

   decrementRepairCounter();
}


// Update the data structure following calls to setPmf().
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
updatePmf() {
   // If the efficiency is low (or if it is time for a repair) try a
   // relatively inexpensive repair.
   if (computeEfficiency() < getMinimumEfficiency() || shouldRepair()) {
      repair();
      // If that didn't do the trick, do a rebuild.
      if (computeEfficiency() < getMinimumEfficiency()) {
         rebuild();
      }
   }
}




// Repair the data structure.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
repair() {
#if 0
   static std::size_t count = 1;
   std::cout << "repair " << count++ << "\n";
#endif
   //
   // Compute the binned PMF.
   //
   std::fill(_binnedPmf.begin(), _binnedPmf.end(), 0);
   _heightUpperBound = 0;
   // For each probability.
   for (std::size_t i = 0; i != _pmf.size(); ++i) {
      // Split the PMF over a number of bins.
      const Number height = _pmf[i] / (_binIndices[i + 1] - _binIndices[i]);
      for (std::size_t j = _binIndices[i]; j != _binIndices[i + 1]; ++j) {
         _binnedPmf[j] = height;
      }
      // Update the upper bound on the bin height.
      if (height > _heightUpperBound) {
         _heightUpperBound = height;
      }
   }

   // Compute the sum of the PMF.
   _pmfSum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);

   resetRepairCounter();
}


// Print information about the data structure.
template<class Generator>
inline
void
DiscreteGeneratorBinsSplitting<true, Generator>::
print(std::ostream& out) const {
   Base::print(out);
   out << "Minimum efficiency factor = " << _minimumEfficiencyFactor << "\n"
       << "Minimum efficiency = " << _minimumEfficiency << "\n"
       << "Efficiency = " << computeEfficiency() << "\n";
   RepairBase::print(out);
}


} // namespace numerical
}

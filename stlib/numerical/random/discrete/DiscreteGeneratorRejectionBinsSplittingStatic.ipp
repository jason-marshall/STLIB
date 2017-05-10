// -*- C++ -*-

#if !defined(__numerical_random_DiscreteGeneratorRejectionBinsSplittingStatic_ipp__)
#error This file is an implementation detail of DiscreteGeneratorRejectionBinsSplittingStatic.
#endif

namespace stlib
{
namespace numerical {

template<bool ExactlyBalance, class Generator>
inline
typename DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::result_type
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
operator()() {
   // Loop until the point is not rejected.
   for (;;) {
      const unsigned random = (*_discreteUniformGenerator)();
      // Use the first bits for indexing.
      const unsigned binIndex = random & getIndexMask();
      // Use the remaining bits for the height deviate.
      const unsigned heightGenerator = random >> getIndexBits();
      const Number height = heightGenerator * _heightUpperBound *
                            getMaxHeightInverse();
      const std::size_t deviate = _deviateIndices[binIndex];
      // If we have a hit for the PMF in this bin.
      if (height < computeBinHeight(deviate)) {
         return deviate;
      }
   }
}

// Initialize the probability mass function.
template<bool ExactlyBalance, class Generator>
template<typename ForwardIterator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
initialize(ForwardIterator begin, ForwardIterator end) {
   // Build the array for the PMF.
   _pmf.resize(std::distance(begin, end));
   std::copy(begin, end, _pmf.begin());
   _inverseSizes.resize(_pmf.size());
   _sortedPmf.resize(_pmf.size());
   if (ExactlyBalance) {
      _binIndices.resize(_pmf.size());
   }

   // Make sure that there are an adequate number of bins.
   if (_deviateIndices.size() < 2 * _pmf.size()) {
      std::size_t indexBits = 1;
      while (std::size_t(1) << indexBits < 2 * _pmf.size()) {
         ++indexBits;
      }
      setIndexBits(indexBits);
   }
   assert(2 * _pmf.size() <= _deviateIndices.size());

   // Rebuild the data structure by splitting the PMF across the bins.
   rebuild();
}


// Rebuild the bins.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
rebuild() {
  _sum = ext::sum(_pmf);
   packIntoBins();
   // Compute an upper bound on the bin height.
   computeUpperBound();
}


// Pack the blocks into bins.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
_packIntoBins(std::false_type /*ExactlyBalance*/) {
   // Get a sorted array of the PMF.
   computeSortedPmf();

   //
   // Determine how many bins to use for each probability.
   //
   Number sum = _sum;
   int remainingBins = getNumberOfBins();
   // For all except the largest probability.
   for (std::size_t i = 0; i != _sortedPmf.size() - 1; ++i) {
      // Determine how many bins to use for this probability.  We must use
      // at least one.  Below I round to the nearest integer.
      const int count = std::max(1,
                                 int(*_sortedPmf[i] / sum * remainingBins + 0.5));
      // The index of the probability.
      const int index = _sortedPmf[i] - _pmf.begin();
      // The inverse of the number of bins for this probability.
      _inverseSizes[index] = 1.0 / count;
      for (int n = 0; n != count; ++n) {
         _deviateIndices[--remainingBins] = index;
      }
      sum -= _pmf[index];
   }
   // Give the largest probability the remaining bins.
#ifdef STLIB_DEBUG
   assert(remainingBins != 0);
#endif
   // The index of the largest probability.
   const int index = *(_sortedPmf.end() - 1) - _pmf.begin();
   // The inverse of the number of bins for this probability.
   _inverseSizes[index] = 1.0 / remainingBins;
   while (remainingBins != 0) {
      _deviateIndices[--remainingBins] = index;
   }
}


// Pack the blocks into bins.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
_packIntoBins(std::true_type /*ExactlyBalance*/) {
   // Clear the old bins.
   for (std::size_t i = 0; i != _binIndices.size(); ++i) {
      _binIndices[i].clear();
   }

   // Get a sorted array of the PMF.
   computeSortedPmf();

   //
   // Determine how many bins to use for each probability.
   //
   Number sum = _sum;
   std::size_t remainingBins = getNumberOfBins();
   // For all except the largest probability.
   for (std::size_t i = 0; i != _sortedPmf.size() - 1; ++i) {
      // Determine how many bins to use for this probability.  We must use
      // at least one.  Below I round to the nearest integer.
      const std::size_t count =
         std::max(std::size_t(1),
                  std::size_t(*_sortedPmf[i] / sum * remainingBins + 0.5));
      // The index of the probability.
      const std::size_t index = _sortedPmf[i] - _pmf.begin();
      std::vector<std::size_t>& indices = _binIndices[index];
      for (std::size_t n = 0; n != count; ++n) {
         indices.push_back(--remainingBins);
         _deviateIndices[remainingBins] = index;
      }
      sum -= _pmf[index];
   }
   // Give the largest probability the remaining bins.
#ifdef STLIB_DEBUG
   assert(remainingBins != 0);
#endif
   // The index of the largest probability.
   const std::size_t index = *(_sortedPmf.end() - 1) - _pmf.begin();
   std::vector<std::size_t>& indices = _binIndices[index];
   while (remainingBins != 0) {
      indices.push_back(--remainingBins);
      _deviateIndices[remainingBins] = index;
   }

   // Balance to minimize the maximum bin height.
   balance();

   // Compute the inverse of the number of bins for each probability.
   computeInverseSizes();
}


// Compute the inverses.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
computeInverseSizes() {
   for (std::size_t i = 0; i != _binIndices.size(); ++i) {
      _inverseSizes[i] = 1.0 / _binIndices[i].size();
   }
}

// Balance to minimize the maximum bin height.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
balance() {
   // Trade bins until we have minimized the maximum bin height.
   while (tradeBins())
      ;
}

// Trade bins to reduce the maximum bin height.
template<bool ExactlyBalance, class Generator>
inline
bool
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
tradeBins() {
   // Get the maximum bin height and the minimum height obtained by removing
   // a bin.
   Number maximumHeight = 0;
   Number minimumHeight = std::numeric_limits<Number>::max();
   Number height;
   std::size_t maximumIndex = std::numeric_limits<std::size_t>::max(),
               minimumIndex = std::numeric_limits<std::size_t>::max();
   for (std::size_t i = 0; i != _pmf.size(); ++i) {
      height = _pmf[i] / _binIndices[i].size();
      if (height > maximumHeight) {
         maximumHeight = height;
         maximumIndex = i;
      }
      if (_binIndices[i].size() != 1) {
         height = _pmf[i] / (_binIndices[i].size() - 1);
         if (height < minimumHeight) {
            minimumHeight = height;
            minimumIndex = i;
         }
      }
   }
   // If we can reduce the maximum height by trading bins.
   if (minimumHeight < maximumHeight && minimumIndex != maximumIndex) {
      _deviateIndices[_binIndices[minimumIndex].back()] = maximumIndex;
      _binIndices[maximumIndex].push_back(_binIndices[minimumIndex].back());
      _binIndices[minimumIndex].pop_back();
      return true;
   }
   return false;
}

// Print information about the data structure.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator>::
print(std::ostream& out) const {
   out << "Bin data:\n\n"
       << "Height upper bound = " << _heightUpperBound << "\n"
       << "Deviate indices = \n" << _deviateIndices << "\n"
       << "\nPMF data:\n\n"
       << "PMF sum = " << _sum << "\n"
       << "PMF = \n" << _pmf << "\n"
       << "Inverse sizes = \n" << _inverseSizes << "\n"
       << "Bin indices = \n";
   for (std::size_t i = 0; i != _binIndices.size(); ++i) {
      for (std::size_t j = 0; j != _binIndices[i].size(); ++j) {
         out << _binIndices[i][j] << " ";
      }
      out << "\n";
   }
}

} // namespace numerical
}

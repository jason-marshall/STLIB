// -*- C++ -*-

#if !defined(__numerical_random_DiscreteGeneratorRejectionBinsSplitting_ipp__)
#error This file is an implementation detail of DiscreteGeneratorRejectionBinsSplitting.
#endif

namespace stlib
{
namespace numerical {

// Initialize the probability mass function.
template<bool ExactlyBalance, class Generator>
template<typename ForwardIterator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
initialize(ForwardIterator begin, ForwardIterator end) {
   Base::initialize(begin, end);
   // The initial error in the sum of the PMF.
   _error = size() * _sum * std::numeric_limits<Number>::epsilon();
   updateMinimumAllowedEfficiency();
}


// Rebuild the bins.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
rebuild() {
   Base::rebuild();
   // The initial error in the sum of the PMF.
   _error = size() * _sum * std::numeric_limits<Number>::epsilon();
   updateMinimumAllowedEfficiency();
}



// Set the probability mass function with the specified index.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
set(const std::size_t index, const Number value) {
   // CONTINUE: Should I remove this?
   // If the value has not changed, do nothing.  I need this check; otherwise
   // the following branch could be expensive.
   if (_pmf[index] == value) {
      return;
   }

   // The remainder of this function is the standard case.  Update the data
   // structure using the difference between the new and old values.

   // Update the error in the PMF sum.
   _error += (_sum + value + _pmf[index]) *
             std::numeric_limits<Number>::epsilon();
   // Update the sum of the PMF.
   _sum += value - _pmf[index];
   // Update the PMF array.
   _pmf[index] = value;

   // Update the upper bound on the bin height.
   const Number height = computeBinHeight(index);
   if (height > _heightUpperBound) {
      _heightUpperBound = height;
   }
}


// Update the data structure following calls to setPmf().
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
update() {
   // The allowed relative error is 2^-32.
   const Number allowedRelativeError = 2.3283064365386963e-10;
   if (_error > allowedRelativeError * _sum) {
      repair();
   }
}


// Repair the data structure.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
repair() {
   // Compute the sum of the PMF.
   Base::computePmfSum();
   // The initial error in the sum.
   _error = size() * _sum * std::numeric_limits<Number>::epsilon();
}


// Print information about the data structure.
template<bool ExactlyBalance, class Generator>
inline
void
DiscreteGeneratorRejectionBinsSplitting<ExactlyBalance, Generator>::
print(std::ostream& out) const {
   Base::print(out);
   out << "Minimum efficiency factor = " << _minimumEfficiencyFactor << "\n"
       << "Minimum efficiency = " << _minimumEfficiency << "\n"
       << "Efficiency = " << computeEfficiency() << "\n";
}

} // namespace numerical
}

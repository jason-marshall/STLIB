// -*- C++ -*-

// Do nothing because the discrete generator automatically update the PMF sum.
template<typename _Generator>
inline
void
updateSum(_Generator* /*generator*/,
          std::true_type /*Automatic update*/) {
}

// Tell the discrete generator to update the PMF sum.
template<typename _Generator>
inline
void
updateSum(_Generator* generator, std::false_type /*Automatic update*/) {
   generator->updateSum();
}

// Update the PMF sum if necessary.
template<typename _Generator>
inline
void
updateSum(_Generator* generator) {
  updateSum(generator, std::integral_constant<bool,
            _Generator::AutomaticUpdate>());
}

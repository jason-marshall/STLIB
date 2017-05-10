// -*- C++ -*-

#if !defined(__sfc_Codes_tcc__)
#error This file is an implementation detail of Codes.
#endif

namespace stlib
{
namespace sfc
{


template<template<typename> class _Order, typename _Traits>
inline
void
checkValidityOfObjectCodes
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes)
{
  for (auto code : codes) {
    // Check that the code itself is valid.
    if (! order.isValid(code)) {
      throw std::runtime_error("Error in stlib::sfc::checkValidityOfObjectCodes(): Invalid code.");
    }
  }
  if (! std::is_sorted(codes.begin(), codes.end())) {
    throw std::runtime_error("Error in stlib::sfc::checkValidityOfObjectCodes(): Codes are not sorted.");
  }
}


template<template<typename> class _Order, typename _Traits>
inline
void
checkValidityOfCellCodes
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes)
{
  // There must at least be a guard cell.
  assert(! codes.empty());
  assert(codes.back() == _Traits::GuardCode);
  for (std::size_t i = 0; i != codes.size() - 1; ++i) {
    // Check that the code itself is valid.
    if (! order.isValid(codes[i])) {
      throw std::runtime_error("Error in stlib::sfc::checkValidityOfCellCodes(): Invalid code.");
    }
    // The codes must be strictly ascending. That is, they are sorted and
    // there are no repeated cells.
    if (! (codes[i] < codes[i + 1])) {
      throw std::runtime_error("Error in stlib::sfc::checkValidityOfCellCodes(): Codes are not strictly ascending.");
    }
  }
}


template<template<typename> class _Order, typename _Traits, typename _T>
inline
void
checkValidityOfCellCodes
(_Order<_Traits> const& order,
 std::vector<std::pair<typename _Traits::Code, _T> > const& pairs)
{
  std::vector<typename _Traits::Code> codes(pairs.size());
  for (std::size_t i = 0; i != codes.size(); ++i) {
    codes[i] = pairs[i].first;
  }
  checkValidityOfCellCodes(order, codes);
}


template<template<typename> class _Order, typename _Traits>
inline
void
checkNonOverlapping
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes)
{
  for (std::size_t i = 0; i != codes.size() - 1; ++i) {
    if (! (codes[i + 1] >= order.location(order.next(codes[i])))) {
      throw std::runtime_error("Error in stlib::sfc::checkNonOverlapping(): "
                               "Codes overlap.");
    }
  }
}


template<template<typename> class _Order, typename _Traits, typename _T>
inline
void
checkNonOverlapping
(_Order<_Traits> const& order,
 std::vector<std::pair<typename _Traits::Code, _T> > const& pairs)
{
  for (std::size_t i = 0; i != pairs.size() - 1; ++i) {
    if (! (pairs[i + 1].first >= order.location(order.next(pairs[i].first)))) {
      throw std::runtime_error("Error in stlib::sfc::checkNonOverlapping(): "
                               "Codes overlap.");
    }
  }
}


} // namespace sfc
} // namespace stlib

// -*- C++ -*-

#ifndef __numerical_integer_print_ipp__
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace numerical
{

// The version for signed integers is intentionally not implemented.

template<typename _Integer>
inline
void
printBitsUnsigned(std::ostream& out, const _Integer x,
                  const std::size_t indexBeginning,
                  const std::size_t indexEnd)
{
#ifdef STLIB_DEBUG
  assert(indexBeginning <= indexEnd &&
         indexEnd <= std::size_t(std::numeric_limits<_Integer>::digits));
#endif
  _Integer mask = 1;
  mask <<= (indexEnd - 1);
  for (std::size_t i = indexBeginning; i != indexEnd; ++i) {
    out << bool(x & mask);
    mask >>= 1;
  }
}

template<typename _Integer>
inline
void
printBits(std::ostream& out, const _Integer x,
          const std::size_t indexBeginning, const std::size_t indexEnd,
          std::false_type /*unsigned*/)
{
  printBitsUnsigned(out, x, indexBeginning, indexEnd);
}


// Interface functions.

template<typename _Integer>
inline
void
printBits(std::ostream& out, const _Integer x)
{
  printBits(out, x, 0, std::numeric_limits<_Integer>::digits);
}


template<typename _Integer>
inline
void
printBits(std::ostream& out, const _Integer x,
          const std::size_t indexBeginning, const std::size_t indexEnd)
{
  printBits(out, x, indexBeginning, indexEnd,
            std::integral_constant<bool,
            std::numeric_limits<_Integer>::is_signed>());
}

} // namespace numerical
}

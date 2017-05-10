// -*- C++ -*-

/**
  \file
  \brief Functions for pair.
*/

#if !defined(__ext_pair_h__)
#define __ext_pair_h__

#include <utility>
#include <iostream>

/// Add using directives for both input and output operators.
#define USING_STLIB_EXT_PAIR_IO_OPERATORS       \
  using stlib::ext::operator<<;                 \
  using stlib::ext::operator>>

/**
\page extPair Extensions to std::pair

Here are functions for \ref extPairFile "file I/O" to extend the functionality
of the std::pair struct [\ref extAustern1999 "Austern, 1999"]. Add 
using directives to use the operators. For an example, below we use
the output operator.
\code
using stlib::ext::operator<<;
std::cout << std::pair<int, int>{2, 3};
\endcode
Below we use a macro to add using directives for both input and output.
\code
USING_STLIB_EXT_PAIR_IO_OPERATORS;
\endcode
*/

namespace stlib
{
namespace ext
{

//----------------------------------------------------------------------------
/// \defgroup extPairFile Pair File I/O
//@{

/// Write the first and second element, separated by a space.
/**
  Format:
  x.first x.second
*/
template<typename _T1, typename _T2>
inline
std::ostream&
operator<<(std::ostream& out, const std::pair<_T1, _T2>& x)
{
  out << x.first << ' ' << x.second;
  return out;
}

/// Read the first and second element.
template<typename _T1, typename _T2>
inline
std::istream&
operator>>(std::istream& in, std::pair<_T1, _T2>& x)
{
  in >> x.first >> x.second;
  return in;
}

//@}

} // namespace ext
} // namespace stlib

#endif

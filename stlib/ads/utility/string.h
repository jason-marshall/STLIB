// -*- C++ -*-

/*!
  \file ads/utility/string.h
  \brief String utility functions.
*/

#if !defined(__ads_utility_string_h__)
#define __ads_utility_string_h__

#include <iomanip>
#include <string>
#include <sstream>

#include <cassert>

namespace stlib
{
namespace ads
{

//-----------------------------------------------------------------------------
/*! \defgroup ads_utility_string Utility: String functions. */
//@{

//! Split the string.  Return the number of words.
template <typename StringOutputIterator>
int
split(const std::string& string, const std::string& separator,
      StringOutputIterator output);

//! Make a zero-padded numerical extension.  Useful for constructing file names.
void
makeZeroPaddedExtension(const int n, int maximumNumber, std::string* ext);

//@}

} // namespace ads
}

#define __ads_utility_string_ipp__
#include "stlib/ads/utility/string.ipp"
#undef __ads_utility_string_ipp__

#endif

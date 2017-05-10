// -*- C++ -*-

/*!
  \file utility.h
  \brief Includes the utility files.
*/

#if !defined(__ads_utility_h__)
#define __ads_utility_h__

#include "stlib/ads/utility/ObjectAndBlankSpace.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

namespace stlib
{
namespace ads
{

/*!
  \page ads_utility Utility Package

  The utility sub-package has the following features.
  - The ParseOptionsArguments class parses command line options
  and arguments.
  - \ref ads_utility_string
  - ObjectAndBlankSpace pads an object with black space.  This is useful in
  avoiding false sharing.
*/

} // namespace ads
} // namespace stlib

#endif

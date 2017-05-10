// -*- C++ -*-

/*!
  \file numerical/integer.h
  \brief Integer utilities.
*/

#if !defined(__numerical_integer_h__)
#define __numerical_integer_h__

#include "stlib/numerical/integer/bits.h"
#include "stlib/numerical/integer/print.h"

namespace stlib
{
namespace numerical
{

/*!
  \page numerical_integer Integer Package.

  - There are utilities for \ref numerical_integer_bits.
  - There are functions that allow you to
  \ref numerical_integer_print "print the bits of an integer".
  - The isNonNegative() function lets you avoid warnings about meaningless 
  comparisons in functions that are templated on an integer type.
*/

} // namespace numerical
} // namespace stlib

#endif

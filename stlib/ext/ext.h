// -*- C++ -*-

#if !defined(__ext_h__)
#define __ext_h__

#include "stlib/ext/array.h"
#include "stlib/ext/pair.h"
#include "stlib/ext/vector.h"

namespace stlib
{
//! Functions that extend the capabilities of the C++ standard library are defined in this namespace.
namespace ext
{
}
}

/*!
\mainpage Standard Library Extensions

\section extIntroduction Introduction

This package provides extensions to the following standard library
classes:
- \ref extArray
- \ref extPair
- \ref extVector

We add functions and operators for these standard library containers in the
stlib::ext namespace. For the functions, use the namespace qualification.
Note that since the function argument(s) will be in the std namespace,
one can't rely on 
<a href="http://en.wikipedia.org/wiki/Argument_dependent_name_lookup">
argument-dependent name lookup</a>.
In the example below we find the minimum
element in a std::vector and calculate the dot product of two
std::vector's.

\code
std::vector<double> x, y;
...
double const minValue = stlib::ext::min(x);
double const d = stlib::ext::dot(x, y);
\endcode

To use the operators, add appropriate using directives. You can specify the
operators individually as follows.
\code
using stlib::ext::operator<<;
\endcode
Alternatively, you can use macros to use all of the operators that are
defined for a class. Below we add using directives for the math operators
for std::array.
\code
USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
\endcode

\section extBibliography Bibliography

-# \anchor extAustern1999
Matthew H. Austern. "Generic Programming and the STL: Using and Extending
the C++ Standard Template Library." Addison-Wesley.
-# \anchor extBecker2007
Pete Becker. "The C++ Standard Library Extensions." Addison-Wesley.
*/

#endif

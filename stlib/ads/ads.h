// -*- C++ -*-

#if !defined(__ads_h__)
#define __ads_h__

#include "stlib/ads/algorithm.h"
#include "stlib/ads/array.h"
#include "stlib/ads/counter.h"
#include "stlib/ads/functor.h"
#include "stlib/ads/halfedge.h"
#include "stlib/ads/indexedPriorityQueue.h"
#include "stlib/ads/iterator.h"
#include "stlib/ads/priority_queue.h"
#include "stlib/ads/set.h"
#include "stlib/ads/tensor.h"
#include "stlib/ads/timer.h"
#include "stlib/ads/utility.h"

//! Sean's template library.
namespace stlib
{
//! The algorithms and data structures package.
namespace ads
{
}
}

/*!
\mainpage Algorithms and Data Structures Package

\section ads_introduction Introduction

This is a templated C++ class library.  All the functionality is
implemented in header files.  Thus there is no library to compile or
link with.  Just include the appropriate header files in your
application code when you compile.

The ADS package is composed of a number of sub-packages.  All classes
and functions are in the \ref ads namespace.  There are some general purpose
sub-packages:
- The \ref ads_priority_queue "priority queue package" has priority queues
implemented with binary heaps and an approximate priority queue implemented
with a cell array.
- The \ref ads_indexedPriorityQueue "indexed priority queue package" has
a variety of data structures.
- The \ref ads_timer "timer package" has a simple timer class.
.
Other sub-packages are preliminary or ad hoc.
- The \ref ads_algorithm "algorithm" package has min and max functions for more
than two arguments and functions for sorting.
- The \ref ads_counter counter package implements the ads::CounterWithReset
class.
- The \ref ads_functor "functor" package defines various utility functors.
- The \ref ads_halfedge "halfedge" package has a halfedge data structure.
- The \ref ads_iterator "iterator" package has iterator adapters.
- The \ref ads_set "set" package has data structures for sets of integers.
- The \ref ads_tensor "tensor" package has square matrices.
- The \ref ads_utility "utility" package has a class for parsing
  command line options, and string functions.
.
Deprecated packages:
- The \ref ads_array "array package" has fixed-size and dynamically-sized
arrays.  There are classes for N-D arrays that either allocate
their own memory, or wrap externally allocated memory. Instead of using these
classes use \c std::array for fixed-size arrays and the top-level
<a href="../container/index.html">array</a> package for dynamically-sized arrays.


\section ads_compiling Compiling


<!---------------------------------
The following compilers are supported:
<TABLE>
<TR>
<TH> Compiler
<TH> Versions
<TH> Flags
<TH> Date Tested
<TH> Notes

<TR>
<TD> GNU Compiler Colection, g++
<TD> 3.4, 4.0, 4.2
<TD> -ansi -pedantic -Wall
<TD> June 3, 2007
<TD>
</TABLE>

- GNU C++ compiler, g++, version 3.3.x, with flags: -ansi -pedantic.
- Intel C++ compiler, icc, version 8.0, with flags: -strict_ansi.
.
This library is ANSI compliant.  (Hence the note about ANSI flags above.
If your code is not ANSI compliant, then you would not want to use
these flags.)  In theory, the code should compile on any platform/compiler
combination that supports ANSI C++.  In practice, I am sure that this is
not the case.  In the future I will test the code with additional
compilers to improve its portability.
--------------------------->

\par
The top-level directory of this package is called \c ads.  Each sub-package
has its own sub-directory.  For example, the array package is in the
\c array directory.  Each sub-package has a header file in the top-level
directory that includes all the classes and functions for that package.
For the timer package, this file is \c timer.h.  To use the timer package,
one would add:
\code
#include "stlib/ads/timer.h"
\endcode
*/

/*!
\page ads_bibliography Bibliography

*/

#endif

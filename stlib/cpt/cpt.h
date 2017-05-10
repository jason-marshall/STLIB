// -*- C++ -*-

/*!
  \file cpt/cpt.h
  \brief The closest point transform to a triangle surface.
*/

//----------------------------------------------------------------------------

#if !defined(__cpt_cpt_h__)
#define __cpt_cpt_h__

#include "stlib/cpt/State.h"

namespace stlib
{
//! All functions and classes in the Closest Transform Package are in the cpt namespace.
namespace cpt {}
}

/*!
\mainpage Closest Point Transform
\anchor cpt

\section cpt_introduction Introduction

This code implements an algorithm for computing the closest point
transform to a triangle mesh surface on a regular 3-D grid.  The closest
point transform finds the Euclidean distance to the triangle mesh.  In
addition, it can compute the closest point on the surface, the closest
triangle face in the mesh and the gradient of the distance.  The
distance, etc., are computed to within a specified distance of the
surface.  The closest point, closest face, distance and gradient of
the distance to the mesh surface are calculated by solving the Eikonal
equation \f$ |\nabla u|^2 = 1 \f$ with the method of characteristics.
The method of characteristics is implemented efficiently with the aid
of computational geometry and polyhedron scan conversion.  The
computed distance is accurate to within machine precision.  The
computational complexity of the algorithm is linear in both the number
of grid points for which the distance is computed and the size of the
mesh.  Thus for many problems, it has the optimal computational complexity.
Visit my web page for publications on solving
static Hamilton-Jacobi equations and in particular for computing the CPT.

\section cpt_compiling Compiling

The following compilers are supported:
<table>
<tr>
<th> Compiler
<th> Versions
<th> Language Flags
<th> Optimization Flags
<th> Date Tested
<th> Notes

<tr>
<td> GCC, g++
<td> 4.0, 4.2
<td> -ansi -pedantic -Wall
<td> -O3 -funroll-loops
<td> June 20, 2007
<td>

<tr>
<td> IBM XL, xlC
<td> September 2004
<td>
<td>
<td> June 30, 2006
<td> Warns that it cannot inline some functions.

<tr>
<td> Intel, icc
<td> 8.0
<td> -strict_ansi
<td> -O3 -Zp16 -ip -ansi_alias
<td> June 3, 2007
<td>

<tr>
<td> PathScale, pathCC
<td> 2.5
<td> -ansi
<td> -O3 -INLINE:aggressive=ON -OPT:alias=typed
<td> June 20, 2007
<td>

<tr>
<td> PGI, pgCC
<td> 7.0
<td>
<td> -O3 -fastsse -Minline
<td> June 20, 2007
<td>
</table>

When using the GNU compilers, you cannot use the \c -fstrict-aliasing flag.
I don't know if this is an issue with my code or the compiler.  I'll try
to resolve this when I get the time.


\section cpt_standard The Standard Interface

The classes State<3,T> and State<2,T>
contain the standard interface (C++ interface) to the CPT package.
If you use the standard interface, there is no library to build.
Just include cpt.h to get the templated class library.

To use the standard interface, instantiate a State<3,T> or State<2,T> class.
Its member functions provide the interface.  Consult the class for this
documentation.

\section cpt_driver The Drivers

\ref cpt_driver2 and \ref cpt_driver3 are example applications that use the
CPT package.  They
read a b-rep from a file and write the distance, the gradient of
the distance, the closest point and the closest face transforms to
files.

*/

#endif

// -*- C++ -*-

/*!
  \file hj/hj.h
  \brief N-D Hamilton-Jacobi interface functions.
*/

//----------------------------------------------------------------------------

#if !defined(__hj_hj_h__)
#define __hj_hj_h__

#include "stlib/hj/DiffSchemeAdj.h"
#include "stlib/hj/DiffSchemeAdjDiag.h"

#include "stlib/hj/DistanceAdj1st.h"
#include "stlib/hj/DistanceAdj2nd.h"
#include "stlib/hj/DistanceAdjDiag1st.h"
#include "stlib/hj/DistanceAdjDiag2nd.h"

#include "stlib/hj/EikonalAdj1st.h"
#include "stlib/hj/EikonalAdj2nd.h"
#include "stlib/hj/EikonalAdjDiag1st.h"
#include "stlib/hj/EikonalAdjDiag2nd.h"

#include "stlib/hj/GridBFS.h"
#include "stlib/hj/GridFM_BH.h"
#include "stlib/hj/GridFM_BHDK.h"
#include "stlib/hj/GridMCC.h"
#include "stlib/hj/GridMCC_CA.h"
#include "stlib/hj/GridSort.h"

#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/grid/RegularGrid.h"

namespace stlib
{
//! All classes and functions in the Hamilton-Jacobi package are in the hj namespace.
namespace hj {

/*! \defgroup utility Utility Functions
  All of the functions take the solution array as an argument.
*/
// @{

//! Initialize the array.
/*!
  Set all the values in the array to infinity.
*/
template<std::size_t N, typename T>
inline
void
initialize(container::MultiArrayRef<T, N>& array) {
   std::fill(array.begin(), array.end(), std::numeric_limits<T>::max());
}

//! Compute the unsigned distance from known values.
/*!
  \param array is the solution array.  All unknown values should be
  infinity.  The distance will be computed from known values.
  \param dx is the grid spacing.
  \param maximumDistance: The distance will be computed up to maximumDistance.
  If \c maximumDistance is zero, the distance will be computed for the
  entire grid.  The default value of \c maximumDistance is zero.
*/
template<std::size_t N, typename T>
void
computeUnsignedDistance(container::MultiArrayRef<T, N>& array,
                        T dx, T maximumDistance = 0);

//! Compute the signed distance from known values.
/*!
  \param array is the solution array.  All unknown values should be
  infinity.  The distance will be computed from known values.
  \param dx is the grid spacing.
  \param maximumDistance: The distance will be computed up to +-maximumDistance.
  If \c maximumDistance is zero, the distance will be computed for the
  entire grid.  The default value of \c maximumDistance is zero.
*/
template<std::size_t N, typename T>
void
computeSignedDistance(container::MultiArrayRef<T, N>& array, T dx,
                      T maximumDistance = 0);


//! Flood fill the unsigned distance.
/*!
  All unknown distances and distances greater than \c maximumDistance are
  set to \c fillValue.  The default value of \c fillValue is
  \c maximumDistance.
*/
template<std::size_t N, typename T>
void
floodFillUnsignedDistance(container::MultiArrayRef<T, N>& array,
                          T maximumDistance, T fillValue = 0);

//! Flood fill the signed distance.
/*!
  All unknown distances and distances greater than \c maximumDistance are
  set to \c +-fillValue.  By default, \c fillValue is set to
  \c maximumDistance.  If there are no known distances, set all the grid points
  to +fillValue.
*/
template<std::size_t N, typename T>
void
floodFillSignedDistance(container::MultiArrayRef<T, N>& array,
                        T maximumDistance, T fillValue = 0);


//! Convert a level set to the signed distance function from the iso-surface.
/*!
  \param array is the field array.  All the values must be finite.
  \param dx is the grid spacing.
  \param maximumDistance is the maximum distance to compute the signed
  distance.  If \c maximumDistance is zero the distance will be computed
  for the entire grid.  The default value of \c maximumDistance is zero.
  \param isoValue is the value of the iso-surface.  The default value
  is zero.
  \param fillValue: If the distance is not computed for the entire
  grid, the point far away from the surface will be flood filled with
  \c +-fillValue.  By default, \c fillValue is set to \c maximumDistance.
*/
template<std::size_t N, typename T>
void
convertLevelSetToSignedDistance(container::MultiArrayRef<T, N>& array, T dx,
                                T isoValue = 0,
                                T maximumDistance = 0,
                                T fillValue = 0);


// CONTINUE: Add an interface that is efficient for many fields.
//! Constant advection of the fields.
/*!
  \param field The field on with to adject values.

  \param grid is the computational grid which holds the grid extents and the
  Cartesian domain.

  \param distance is the distance array.

  \param maximumDistance  The signed distance must have been computed up to
  at least \c maximumDistance.  The boundary condition will be set for grid
  points with distances in the range \f$(-maximumDistance .. 0)\f$.

  \param defaultValue  In the field, grid points with distances in the
  range \f$(- \infty .. -maximumDistance]\f$ will be set to \c defaultValue.
*/
template<typename T, typename F>
void
advectConstantIntoNegativeDistance
(container::MultiArrayRef<F, 3>& field,
 const geom::RegularGrid<3, T>& grid,
 const container::MultiArrayConstRef<T, 3>& distance,
 T maximumDistance,
 F defaultValue);

// @}















/*! \defgroup utility_pointer Utility Functions: Pointer Interface
  These functions take a pointer as an argument.
*/
// @{


//! Initialize the array.
/*!
  Wrapper for
  initialize(container::MultiArrayRef<T,2>& array)
*/
template<typename T>
inline
void
initialize(const std::size_t extentX, const std::size_t extentY, T* data) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   initialize(container::MultiArrayRef<T, 2>(extents, data));
}


//! Initialize the array.
/*!
  Wrapper for
  initialize(container::MultiArrayRef<T,3>& array)
*/
template<typename T>
inline
void
initialize(const std::size_t extentX, const std::size_t extentY,
           const std::size_t extentZ, T* data) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   initialize(container::MultiArrayRef<T, 3>(extents, data));
}




//! Compute the unsigned distance from known values.
/*!
  Wrapper for
  computeUnsignedDistance(container::MultiArrayRef<T,2>& array,T dx,T maximumDistance)
*/
template<typename T>
inline
void
computeUnsignedDistance(const std::size_t extentX, const std::size_t extentY,
                        T* data, const T dx, const T maximumDistance = 0) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   computeUnsignedDistance(container::MultiArrayRef<T, 2>(extents, data),
                           dx, maximumDistance);
}


//! Compute the unsigned distance from known values.
/*!
  Wrapper for
  computeUnsignedDistance(container::MultiArrayRef<T,3>& array,T dx,T maximumDistance)
*/
template<typename T>
inline
void
computeUnsignedDistance(const std::size_t extentX, const std::size_t extentY,
                        const std::size_t extentZ,
                        T* data, const T dx, const T maximumDistance = 0) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   computeUnsignedDistance(container::MultiArrayRef<T, 3>(extents, data),
                           dx, maximumDistance);
}



//! Compute the signed distance from known values.
/*!
  Wrapper for
  computeSignedDistance(container::MultiArrayRef<T,2>& array,T dx,T maximumDistance)
*/
template<typename T>
inline
void
computeSignedDistance(const std::size_t extentX, const std::size_t extentY,
                      T* data, const T dx, const T maximumDistance = 0) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   computeSignedDistance(container::MultiArrayRef<T, 2>(extents, data),
                         dx, maximumDistance);
}


//! Compute the signed distance from known values.
/*!
  Wrapper for
  computeSignedDistance(container::MultiArrayRef<T,3>& array,T dx,T maximumDistance)
*/
template<typename T>
inline
void
computeSignedDistance(const std::size_t extentX, const std::size_t extentY,
                      const std::size_t extentZ,
                      T* data, const T dx, const T maximumDistance = 0) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   computeSignedDistance(container::MultiArrayRef<T, 3>(extents, data),
                         dx, maximumDistance);
}



//! Flood fill the unsigned distance.
/*!
  Wrapper for
  floodFillUnsignedDistance(container::MultiArrayRef<T,2>& array,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
floodFillUnsignedDistance(const std::size_t extentX, const std::size_t extentY,
                          T* data,
                          const T maximumDistance, const T fillValue = 0) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   floodFillUnsignedDistance(container::MultiArrayRef<T, 2>(extents, data),
                             maximumDistance, fillValue);
}


//! Flood fill the unsigned distance.
/*!
  Wrapper for
  floodFillUnsignedDistance(container::MultiArrayRef<T,3>& array,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
floodFillUnsignedDistance(const std::size_t extentX, const std::size_t extentY,
                          const std::size_t extentZ, T* data,
                          const T maximumDistance, const T fillValue = 0) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   floodFillUnsignedDistance(container::MultiArrayRef<T, 3>(extents, data),
                             maximumDistance, fillValue);
}



//! Flood fill the signed distance.
/*!
  Wrapper for
  floodFillSignedDistance(container::MultiArrayRef<T,2>& array,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
floodFillSignedDistance(const std::size_t extentX, const std::size_t extentY,
                        T* data,
                        const T maximumDistance, const T fillValue = 0) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   floodFillSignedDistance(container::MultiArrayRef<T, 2>(extents, data),
                           maximumDistance, fillValue);
}


//! Flood fill the signed distance.
/*!
  Wrapper for
  floodFillSignedDistance(container::MultiArrayRef<T,3>& array,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
floodFillSignedDistance(const std::size_t extentX, const std::size_t extentY,
                        const std::size_t extentZ, T* data,
                        const T maximumDistance, const T fillValue = 0) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   floodFillSignedDistance(container::MultiArrayRef<T, 3>(extents, data),
                           maximumDistance, fillValue);
}



//! Convert a level set to the signed distance function from the iso-surface.
/*!
  Wrapper for
  convertLevelSetToSignedDistance(container::MultiArrayRef<T,2>& array,T dx,T isoValue,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
convertLevelSetToSignedDistance(const std::size_t extentX,
                                const std::size_t extentY,
                                T* data,
                                const T dx,
                                const T isoValue = 0,
                                const T maximumDistance = 0,
                                const T fillValue = 0) {
   const std::array<std::size_t, 2> extents = {{extentX, extentY}};
   convertLevelSetToSignedDistance(container::MultiArrayRef<T, 2>(extents, data),
                                   dx, isoValue, maximumDistance, fillValue);
}


//! Convert a level set to the signed distance function from the iso-surface.
/*!
  Wrapper for
  convertLevelSetToSignedDistance(container::MultiArrayRef<T,3>& array,T dx,T isoValue,T maximumDistance,T fillValue)
*/
template<typename T>
inline
void
convertLevelSetToSignedDistance(const std::size_t extentX,
                                const std::size_t extentY,
                                const std::size_t extentZ, T* data,
                                const T dx,
                                const T isoValue = 0,
                                const T maximumDistance = 0,
                                const T fillValue = 0) {
   const std::array<std::size_t, 3> extents =
      {{extentX, extentY, extentZ}};
   convertLevelSetToSignedDistance(container::MultiArrayRef<T, 3>(extents, data),
                                   dx, isoValue, maximumDistance, fillValue);
}



// @}

//=============================================================================
//=============================================================================
/*!
  \mainpage N-D Static Hamilton-Jacobi Solver

  \section introduction Introduction

  This package contains classes for solving static Hamilton-Jacobi
  equations.  There is an interface in
  hj.h that allows you to solve some common problems with simple
  function calls.

  \section interface Interface for Common Problems

  \subsection functions Functions

  The interface in \ref hj.h provides \ref utility "functions"
  for computing signed and unsigned distance.
  All functions are in the \c hj namespace.

  - hj::initialize()
  - hj::unsignedDistance()
  - hj::signedDistance()
  - hj::floodFillUnsignedDistance()
  - hj::floodFillSignedDistance()
  - hj::levelSetToSignedDistance()
  - hj::constantAdvection()

  There are also \ref utility_pointer "wrapper functions" which take a pointer
  to the array data as an argument.

  \subsection compiling Compiling

  There is no library, just header files.  Thus, to compile your application,
  include the file \c hj.h in your source.

  The H-J package requires my algorithms and data structures (ADS) package.
  The ADS package is
  a templated class library.  Place the ADS package in a convenient directory
  and include the relevant directory in your makefile by using the -I
  compiler flag.

  I have compiled the library using g++ (GCC) 3.4.3.  If you use a different
  compiler or version, the code may need modification.

  \subsection examples Examples

  There are examples of using the interface functions in the test directory.





  \section classes Classes for Solving Hamilton-Jacobi Equations in N-D

  \subsection methods Solution Methods

  The package provides two marching methods for solving static
  Hamilton-Jacobi equations.  Sethian's fast marching method is
  implemented in the hj::GridFM class.  The marching with a
  correctness criterion algorithm is implemented in hj::GridMCC.  In
  addition, there is a method which schedules the labeling operations
  using a breadth first search.  This does not produce the correct
  solution, but gives a lower bound on the execution time of an ideal
  method.  This is implemented in hj::GridBFS.

  \subsection equations Equations

  Currently, one can solve either the homogeneous eikonal equation:
  \f$ | \nabla u | = 1 \f$ or the inhomogeneous eikonal equation:
  \f$ | \nabla u | f = 1 \f$.  Here \f$ f \f$ is the speed function,
  which is specified on the computational grid.  The solution of
  the former equation, \f$ | \nabla u | = 1 \f$, with the boundary
  condition \f$ u|_S = 0 \f$ is the unsigned distance from the
  manifold \f$ S \f$.  This equation is implemented in
  hj::Distance and the classes that derive from it.

  For the problem: \f$ | \nabla u | f = 1 \f$, \f$ u|_S = 0 \f$,
  the solution is the arrival time of a wave propagating with speed
  \f$ f \f$ and starting at the manifold \f$ S \f$.  This equation
  is implemented in hj::Eikonal and the classes that derive from it.

  One can add other equations by providing the functionality in either
  of the above classes.

  \subsection drivers Drivers

  The performance directory has drivers which show how to use the solvers.
  Compile the executables with "make".
  There are programs (for \ref hj_timer2 "2-D" and \ref hj_timer3 "3-D")
  which measure the execution time of any of the three
  methods (fast marching, marching with a correctness criterion or placebo)
  with either the homogeneous or inhomogeneous eikonal equation.
  There are also programs (for \ref hj_error2 "2-D" and \ref hj_error3 "3-D")
  which measure the error in solving the homogeneous
  eikonal equation to compute distance.
*/










//=============================================================================
//=============================================================================
/*!
\page hj_introduction Introduction



<!----------------------------------------------------------------------
-------------------------------COMMENTED OUT----------------------------
The eikonal equation, \f$|\nabla u| f = 1\f$ cite{sethian:1999},
can describe the arrival time of a wave
propagating with speed \f$f\f$.  If \f$f = 1\f$ and the boundary condition is
\f$u|_S = 0\f$, then the solution \f$u\f$ is the distance from the manifold \f$S\f$.
In the method of characteristics solution of the distance problem,
the characteristics
are straight lines which are orthogonal to \f$S\f$.  We call the direction in
which the characteristics propagate the \e downwind direction.
More than one characteristic may reach a given point.  In this case the
solution is multi-valued.  One can obtain a single-valued weak solution
by choosing the smallest of the multi-valued solutions at each point.
This is a weak solution because \f$u\f$ is continuous, but not everywhere
differentiable.
By analogy to fluid flow problems, this weak solution is called the
<em>viscosity solution</em>.

One can use finite difference methods to numerically solve the eikonal
equation on a grid cite{osher/sethian:1988}.
In order to converge, the differences must be
taken in the upwind direction.  That is, smaller values of distance
are used to determine the larger values.  This models how a wave
propagates from the initial position outward, following the
characteristics in the downwind direction.  The finite difference
scheme can be solved iteratively by applying it to each grid point
until the solution converges.  However, the number of iterations required
is typically \f$\mathcal{O}(N^{1/d})\f$ in a \f$d\f$-dimensional grid with \f$N\f$
grid points.  The scheme may be efficiently and directly solved by
ordering the grid points so that information is always propagated in
the direction of increasing distance.  This is <em>Sethian's Fast
Marching Method</em> cite{sethian:1996} cite{sethian:1999}.
It achieves a computational
complexity of \f$\mathcal{O}(N \log N)\f$.
--------------------------------------------------------------------->


We will first describe upwind difference schemes and
the Fast Marching Method (FMM) for solving static Hamilton-Jacobi
equations.  Then we will develop a Marching with a Correctness
Criterion (MCC) algorithm for solving this problem.  We will find that
the MCC algorithm requires nonstandard upwind finite difference schemes.  We
will show that the MCC algorithm produces the same solution as the FMM,
but can have computational complexity \f$\mathcal{O}(N)\f$, the optimal
complexity for this problem.  We will perform tests to demonstrate the
linear complexity of the MCC algorithm and to compare its performance
to that of the FMM.
*/








//=============================================================================
//=============================================================================
/*!
\page hj_upwind Upwind Finite Difference Schemes



In this section we will present a first-order and a second-order scheme for
solving the eikonal equation.  Here we provide a short summary of material in
cite{sethian:1999}.
We consider solving the eikonal equation, \f$|\nabla u| f = 1\f$ in 2-D.
Let \f$u_{i,j}\f$ be the
approximate solution on a regular grid with spacing \f$\Delta x\f$.  We define
one-sided difference operators which provide first-order approximations
of \f$\partial u / \partial x\f$ and \f$\partial u / \partial y\f$.
\f[
D_{i,j}^{+x} u = \frac{u_{i+1,j} - u_{i,j}}{\Delta x}, \quad
D_{i,j}^{-x} u = \frac{u_{i,j} - u_{i-1,j}}{\Delta x}
\f]
\f[
D_{i,j}^{+y} u = \frac{u_{i,j+1} - u_{i,j}}{\Delta x}, \quad
D_{i,j}^{-y} u = \frac{u_{i,j} - u_{i,j-1}}{\Delta x}
\f]
Suppose \f$u_{i,j}\f$ is the approximate solution.  To compute
\f$\partial u / \partial x\f$ at the grid point \f$(i,j)\f$,
we will use differencing in the upwind direction.
If \f$u_{i-1,j} < u_{i,j} < u_{i+1,j}\f$ then the left is an
upwind direction and
\f$\partial u / \partial x \approx D_{i,j}^{-x}\f$.
If \f$u_{i-1,j} > u_{i,j} > u_{i+1,j}\f$ then the right is an
upwind direction and
\f$\partial u / \partial x \approx D_{i,j}^{+x}\f$.
If both \f$u_{i-1,j}\f$ and \f$u_{i+1,j}\f$ are less than \f$u_{i,j}\f$ then we
determine the upwind direction by choosing the smaller of the two.
If both \f$u_{i-1,j}\f$ and \f$u_{i+1,j}\f$ are greater than \f$u_{i,j}\f$
then there is no upwind direction.
The derivative in the \f$x\f$ direction vanishes.
We can concisely encode this information into the
<em>first-order, adjacent difference scheme</em>:
\f[
  \left(
    \left( \mathrm{max} \left( D_{i,j}^{-x} u, - D_{i,j}^{+x}, 0 \right) \right)^2
    + \left( \mathrm{max} \left( D_{i,j}^{-y} u, - D_{i,j}^{+y}, 0 \right) \right)^2
  \right)^{1/2} = \frac{1}{f_{i,j}}
\f]
If the four adjacent neighbors of the grid point \f$u_{i,j}\f$ are known, then
the difference scheme gives a quadratic equation for \f$u_{i,j}\f$.

Now we seek a second-order accurate scheme.
If \f$u_{i-2,j} < u_{i-1,j} < u_{i,j}\f$ then the left is an upwind direction.
We use the two adjacent grid points to the left to get a second-order accurate
approximation of \f$\partial u / \partial x\f$.
\f[
\frac{\partial u}{\partial x} \approx \frac{ 3 u_{i,j} - 4 u_{i-1,j} + u_{i-2,j} }{ 2 \Delta x }
\f]
If \f$u_{i-2,j} > u_{i-1,j} < u_{i,j}\f$ then the left is still an upwind direction at
\f$(i,j)\f$, however we only use the closest adjacent grid point
in the difference scheme.  Thus we are limited to a first-order difference
scheme in the left direction.

We can write the second-order accurate formula in terms of the
one-sided differences.
\f[
\frac{\partial u}{\partial x} \approx D_{i,j}^{-x} u +
\frac{\Delta x}{2} D_{i,j}^{-x} D_{i,j}^{-x} u
\f]
By defining the switch function \f$s_{i,j}^{-x}\f$ we can write a
formula that is second-order accurate when
\f$u_{i-2,j} < u_{i-1,j} < u_{i,j}\f$ and reverts to the
first-order formula when \f$u_{i-2,j} > u_{i-1,j} < u_{i,j}\f$.
\f[
  s_{i,j}^{-x} = \begin{cases}
    1 & \mathrm{if } u_{i-2,j} < u_{i-1,j} \\
    0 & \mathrm{otherwise}
    \end{cases}
\f]
\f[
\frac{\partial u}{\partial x} \approx D_{i,j}^{-x} u + s_{i,j}^{-x} \frac{\Delta x}{2} D_{i,j}^{-x} D_{i,j}^{-x} u
\f]
We use this to make a second-order accurate finite difference scheme:
\f[
\left(
  \left( \mathrm{max} \left( D_{i,j}^{-x} u + s_{i,j}^{-x} \frac{\Delta x}{2} D_{i,j}^{-x} D_{i,j}^{-x} u,
      - \left( D_{i,j}^{+x} - s_{i,j}^{+x} \frac{\Delta x}{2} D_{i,j}^{+x} D_{i,j}^{+x} u \right), 0 \right)
  \right)^2 \right.
\f]
\f[
  + \left. \left( \mathrm{max} \left( D_{i,j}^{-y} u + s_{i,j}^{-y} \frac{\Delta x}{2} D_{i,j}^{-y} D_{i,j}^{-y} u,
      - \left( D_{i,j}^{+y} - s_{i,j}^{+y} \frac{\Delta x}{2} D_{i,j}^{+y} D_{i,j}^{+y} u \right), 0 \right)
  \right)^2
\right)^{1/2} = \frac{1}{f_{i,j}}
\f]
If the adjacent neighbors of the grid point \f$u_{i,j}\f$ are known, then
this gives a quadratic equation for \f$u_{i,j}\f$.
*/











//=============================================================================
//=============================================================================
/*!
\page hj_fast The Fast Marching Method


Tsitsiklis was the first to publish a single-pass algorithm for
solving static Hamilton-Jacobi equations.  Addressing a trajectory
optimization problem, he presented a first-order accurate
<em>Dijkstra-like algorithm</em> in cite{tsitsiklis:1995}.
Sethian discovered the method independently and published
his <em>Fast Marching Method</em> in cite{sethian:1996}.
He presented higher-order accurate methods and applications in
cite{sethian:SIAM:1999}.


The Fast Marching Method is similar to Dijkstra's algorithm
cite{cormen:2001} for computing the single-source shortest paths in a
weighted, directed graph.  In solving this problem, each vertex is
assigned a distance, which is the sum of the edge weights along the
minimum-weight path from the source vertex.  As Dijkstra's algorithm
progresses, the status of each vertex is either known, labeled or
unknown.  Initially, the source vertex in the graph has known status
and zero distance.  All other vertices have unknown status and
infinite distance.  The source vertex labels each of its
adjacent neighbors.  A known vertex labels an adjacent vertex by
setting its status to labeled if it is unknown and setting its
distance to be the minimum of its current distance and the sum of the
known vertices' weight and the connecting edge weight.  It can be
shown that the labeled vertex with minimum distance has the correct
value.  Thus the status of this vertex is set to known, and it labels its
neighbors.  This process of freezing the value of the minimum labeled
vertex and labeling its adjacent neighbors is repeated until no
labeled vertices remain.  At this point all the vertices that are
reachable from the source have the correct shortest path distance.
The performance of Dijkstra's algorithm depends on being able to
quickly determine the labeled vertex with minimum distance.  One can
efficiently implement the algorithm by storing the labeled vertices in
a binary heap.  Then the minimum labeled vertex can be determined in
\f$\mathcal{O}(\log n)\f$ time where \f$n\f$ is the number of labeled
vertices.

Sethian's Fast Marching Method differs from Dijkstra's algorithm in that
the finite difference scheme is used to label the adjacent neighbors
when a grid point becomes known.  If there are \f$N\f$ grid points, the
labeling operations have a computational cost of \f$\mathcal{O}(N)\f$.
Since there may be at most \f$N\f$ labeled grid points, maintaining the
binary heap and choosing the minimum labeled grid points adds a cost
of \f$\mathcal{O}(N \log N)\f$.  Thus the total complexity is
\f$\mathcal{O}(N \log N)\f$.


We consider how a grid point that has become known labels its adjacent
neighbors using the first-order, adjacent, upwind difference scheme.
For each adjacent neighbor,
there are potentially three ways to compute a new solution there.
In the figure below, the center grid point has just
become known.  Suppose the adjacent neighbor to the right is not known.
We show three ways the center grid point can be used
to compute the value of this neighbor.  First, only the single known
grid point is used.  This corresponds to the case that there is no
vertical upwind direction.  If the grid points that are diagonal to the
known grid point and adjacent to the grid point being labeled are known,
they can be used in the difference scheme as well.  This accounts for
the second two cases.


\image html LabelAdjacent.jpg "The three ways of labeling an adjacent neighbor using the first-order, adjacent difference scheme."
\image latex LabelAdjacent.pdf "The three ways of labeling an adjacent neighbor using the first-order, adjacent difference scheme." width=0.9\textwidth


Below we give the functions that implement the first-order, adjacent
difference scheme for the eikonal equation, \f$|\nabla u| f = 1\f$.  The difference
scheme can use a single adjacent grid point (\c differenceAdj())
or two adjacent grid points (\c differenceAdjAdj()).  The
second of these solves a quadratic equation to determine the solution.
After we compute the solution in \c differenceAdjAdj(),
we check that the solution is not less than its two known neighbors.
If the solution is less than one of its neighbors, then the scheme
would not be upwind.  In this case, we return infinity.


\verbatim
differenceAdj(a):
  return a + dx / f \endverbatim


\verbatim
differenceAdjAdj(a, b):
  discriminant = 2 dx^2 / f^2 - (a - b)^2
  if discriminant >= 0:
    solution = (a + b + sqrt(discriminant)) / 2
    if solution >= a and solution >= b:
      return solution
  return Infinity \endverbatim


We give a more efficient method of implementing \c differenceAdjAdj()
below.  If the condition in \c differenceAdjAdj() is not satisfied,
then the characteristic line comes from outside the wedge defined by
the two adjacent neighbors.  In this case, the computed value will be
higher than one of \c a or \c b and thus the difference
scheme will not be upwind.  For this case, we return infinity.


\verbatim
differenceAdjAdj(a, b):
  if |a - b| <= dx / f:
    return (a + b + sqrt(2 dx^2 / f^2 - (a - b)^2 )) / 2
  return Infinity \endverbatim


Below is the Fast Marching Method for a 2-D grid.
As input it takes a grid with a solution
array and a status array.   The initial condition has been specified by setting
the status at some grid points to KNOWN and setting the solution there.
At all other grid points the status is UNLABELED and the solution is \f$\infty\f$.
The binary heap which stores the labeled grid points supports
three operations:
- \c push():
  Grid points are added to the heap when they become labeled.
\c extractMinimum():
  The grid point with minimum solution can be removed.  This function returns
  the indices of that grid point.
\c decrease():
  The solution at a grid point in the labeled set may be decreased through
  labeling.  This function adjusts the position of the grid point in the heap.
.
The binary heap takes the solution array as an argument in its constructor
because it stores pointers into this array.


\verbatim
fastMarching(grid):
  // Make the binary heap.
  BinaryHeap labeled(grid.solution)
  // Label the neighbors of known grid points.
  for each (i, j):
    if grid.status(i, j) == KNOWN:
      grid.labelNeighbors(labeled, i, j)
  // Loop until there are no labeled grid points.
  while labeled is not empty:
    (i, j) = labeled.extractMinimum()
    grid.labelNeighbors(labeled, i, j)
  return \endverbatim


Below is the \c labelNeighbors() function which uses the finite
difference scheme to label the neighbors.  This function uses the first-order,
adjacent scheme.
Thus it labels the four adjacent neighbors.  The
\c label() function updates the value of a grid point and manages
the heap of labeled grid points.


\verbatim
labelNeighbors(grid, labeled, i, j):
  grid.status(i, j) = KNOWN
  soln = grid.solution(i, j)
  for the four adjacent indices (p, q):
    adjSoln = differenceAdj(soln)
    (m, n) = indices diagonal to (i, j) and adjacent to (p, q)
    if grid.status(m, n) == KNOWN:
      adjSoln = min(adjSoln, differenceAdjAdj(soln, grid.solution(m, n)))
    (m, n) = other indices diagonal to (i, j) and adjacent to (p, q)
    if grid.status(m, n) == KNOWN:
      adjSoln = min(adjSoln, differenceAdjAdj(soln, grid.solution(m, n)))
    label(grid, p, q, adjSoln)
  return \endverbatim


\verbatim
label(grid, labeled, i, j, value):
  if grid.status(i, j) == UNLABELED:
    grid.status(i, j) = LABELED
    grid.solution(i, j) = value
    labeled.push(i, j)
  else if grid.status(i, j) == LABELED and value < grid.solution(i, j):
    grid.solution(i, j) = value
    labeled.decrease(i, j)
  return \endverbatim


<!---------------------------------------------------------------------------->
\section hj_fast_status The Status Array

One can implement the Fast Marching Method without the use of the
status array.  In this case one does not check that a grid point is
known when using it to label neighbors.  A solution value of \f$\infty\f$
signifies that a grid point has not been labeled.  Below is this
variation of the Fast Marching Method.


\verbatim
fastMarchingNoStatus(grid):
  // Make the binary heap.
  BinaryHeap labeled(grid.solution)
  // Label the neighbors of known grid points.
  for each (i, j):
    if grid.solution(i, j) != Infinity:
      grid.labelNeighbors(labeled, i, j)
  // Loop until there are no labeled grid points.
  while labeled is not empty:
    (i, j) = labeled.top()
    labeled.pop()
    grid.labelNeighbors(labeled, i, j)
  return \endverbatim


\verbatim
labelNeighbors(grid, labeled, i, j)
  soln = grid.solution(i, j)
  for the four adjacent indices (p, q):
    adjSoln = differenceAdj(soln)
    (m, n) = indices diagonal to (i, j) and adjacent to (p, q)
    if grid.solution(m, n) != Infinity:
      adjSoln = min(adjSoln, differenceAdjAdj(soln, grid.solution(m, n)))
    (m, n) = other indices diagonal to (i, j) and adjacent to (p, q)
    if grid.solution(m, n) != Infinity:
      adjSoln = min(adjSoln, differenceAdjAdj(soln, grid.solution(m, n)))
    label(grid, p, q, adjSoln)
  return \endverbatim


\verbatim
label(grid, labeled, i, j, value)
  if grid.solution(i, j) == Infinity:
    grid.solution(i, j) = value
    labeled.push(i, j)
  else if value < grid.solution(i, j):
    grid.solution(i, j) = value
    labeled.decrease(i, j)
  return \endverbatim


Forgoing the use of the status array decreases the memory requirement but
increases the execution time.  This is because the finite difference scheme
is called more often.  In the figure below we show the execution
times for solving the eikonal equation \f$|\nabla u| = 1\f$ on a 2-D
grid with a first-order, adjacent difference scheme.  For this test, not
using a status array increased the execution time by about \f$50\%\f$.  All
further implementations of the Fast Marching Method presented in this chapter
use a status array.


\image html Status.jpg "Log-log plots of the execution time per grid point versus the number of grid points for the fast marching method with and without a status array."
\image latex Status.pdf "Log-log plots of the execution time per grid point versus the number of grid points for the fast marching method with and without a status array." width=0.6\textwidth
*/











//=============================================================================
//=============================================================================
/*!
\page hj_mcc Applying the Marching with a Correctness Criterion Method


In analogy with with the single-source shortest paths problem,
we try to apply the Marching with a Correctness
Criterion method to obtain an ordered upwind method of solving static
Hamilton-Jacobi equations.  Below we adapt the MCC algorithm in
the section \ref shortest_paths_greedier .  Only minor changes are
required.


\verbatim
marchingWithCorrectnessCriterion(grid):
  labeled.clear()
  newLabeled.clear()
  // Label the neighbors of known grid points.
  for each (i, j):
    if grid.status(i, j) == KNOWN:
      grid.labelNeighbors(labeled, i, j)
  // Loop until there are no labeled grid points.
  while labeled is not empty:
    for (i,j) in labeled:
      if grid.solution(i, j) is determined to be correct:
        grid.labelNeighbors(newLabeled, i, j)
    // Get the labeled lists ready for the next step.
    removeKnown(labeled)
    labeled += newLabeled
    newLabeled.clear()
  return \endverbatim


The algorithm is very similar to Sethian's Fast Marching Method.  However,
the labeled grid points are stored in an array or list instead of a binary
heap.  When a labeled grid point is determined to have the correct solution,
it uses the difference scheme to label its neighbors and it is removed from
the labeled set.  The \c labelNeighbors() function depends on the
difference scheme used.  The \c label() function is the same
as before, except that there is no need to adjust the position of a labeled
grid point when the solution decreases.


\verbatim
label(grid, labeled, i, j, value)
  if grid.status(i, j) == UNLABELED:
    grid.status(i, j) = LABELED
    grid.solution(i, j) = value
    labeled += (i, j)
  else if grid.status(i, j) == LABELED and value < grid.solution(i, j):
    grid.solution(i, j) = value
  return \endverbatim


We consider two test problems to probe the possible usefulness of a
Marching with a Correctness Criterion algorithm.  We solve the eikonal
equation \f$|\nabla u| = 1\f$ on a \f$5 \times 5\f$ grid.  For the first
problem, we compute distance from a point that coincides with the
lower left grid point.  For the second problem, we compute distance
from a line segment that makes an angle of \f$-5^\circ\f$ with the \f$x\f$
axis.  See the figure below for a diagram of the
grids and initial conditions.  The diagram also shows the directions
from which the characteristic lines come.  For the initial condition
of the first problem, the solution at grid point \f$(0,0)\f$ is set to
zero.  For the second problem, the solution is set at grid points
(0,0) through (4,0).


\image html GridDirectionCircle.jpg "Test problem for exploring the Marching with a Correctness Criterion algorithm: Distance is computed from the (0,0) grid point."
\image latex GridDirectionCircle.pdf "Test problem for exploring the Marching with a Correctness Criterion algorithm: Distance is computed from the (0,0) grid point." width=0.4\textwidth

\image html GridDirectionLine.jpg "Test problem for exploring the Marching with a Correctness Criterion algorithm: Distance is computed from a line segment."
\image latex GridDirectionLine.pdf "Test problem for exploring the Marching with a Correctness Criterion algorithm: Distance is computed from a line segment." width=0.4\textwidth


We consider solving the test problems with a first-order, adjacent
stencil.  That is, a grid point uses its four adjacent neighbors in
the difference scheme.  The figure below shows
a <em>dependency diagram</em> for the two test problems.  That is,
each grid point which is not set in the initial condition has arrows
pointing to the grid points on which it depends (the grid points used
in the difference scheme which produce the correct solution).  Note
that the arrows identify the quadrant from which the characteristic
line comes.  (Except for the degenerate cases where a single adjacent
grid point is used in the difference scheme.)


\image html GridDependency4PtCircle.jpg "Dependency diagram for a 5-point, adjacent stencil."
\image latex GridDependency4PtCircle.pdf "Dependency diagram for a 5-point, adjacent stencil." width=0.4\textwidth

\image html GridDependency4PtLine.jpg "Dependency diagram for a 5-point, adjacent stencil."
\image latex GridDependency4PtLine.pdf "Dependency diagram for a 5-point, adjacent stencil." width=0.4\textwidth


Now we need to develop a correctness criterion and see if it leads to
an efficient algorithm.  We already have one correctness criterion,
namely the one used in the Fast Marching Method:  The labeled grid
point with minimum solution is correct.  Now we develop a more
sophisticated one in analogy with the level 1 correctness criterion
developed for graphs in the section
\ref shortest_paths_greedier .
That is, we will determine a lower bound on the solution at a labeled
grid point using the assumption that it has at least
one unknown neighbor.  If the predicted solution there is no larger
than this lower bound, then it must be correct.

Let \f$\mu\f$ be the minimum solution among the labeled grid points.  Each
labeled grid point has at least one known adjacent neighbor.  The
correct solution at any unknown adjacent neighbors may be as low as
\f$\mu\f$.  We determine the smallest predicted solution using an unknown
neighbor.  The solution at the known neighbor may be as small as
\f$\mu - \Delta\mathsf{x}\f$.  (If it were smaller, the solution at the labeled
grid point would be less than \f$\mu\f$.)  If the known adjacent neighbor
has a value of \f$\mu - \Delta\mathsf{x}\f$ and the unknown adjacent
neighbor has a value of \f$\mu\f$, then the first-order, adjacent scheme
computes a value of \f$\mu\f$ for the labeled grid point.  Thus \f$\mu\f$ is our
lower bound; any labeled grid point with solution less than or equal
to \f$\mu\f$ is correct.  This is disappointing.  Our more sophisticated
correctness criterion is the same as the one used in the Fast Marching
Method: the labeled grid point with smallest solution is correct.

The figures below show <em>order diagrams</em> for
the two test problems when using the above correctness criterion with
the adjacent stencil.  That is, the grid points are labeled with the
order in which they can be computed.  The results are disconcerting in
that the order does not reflect the direction of the characteristic
lines.  The flow of information in the difference scheme is very
different than the flow of information in the analytical solution.
Firstly, the difference scheme itself is troubling.  Consider the
second test problem in which distance is computed from a line segment.
The solution at the grid point \f$(4,1)\f$ depends on all of the grid
points set in the initial condition: \f$(0,0)\f$ through \f$(4,0)\f$.  Yet,
the direction of the characteristic line is nearly vertical implying
that the flow of information should be roughly vertical.  Secondly,
the correctness criterion would not lead to an efficient algorithm.
In the second test problem, only a single labeled grid point is
determined to be correct at each step.  (The symmetry in the first
problem leads to either one or two labeled grid points being
determined at each step.)


\image html GridOrder4PtCircle.jpg "Order diagram for the 5-point, adjacent stencil."
\image latex GridOrder4PtCircle.pdf "Order diagram for the 5-point, adjacent stencil." width=0.4\textwidth

\image html GridOrder4PtLine.jpg "Order diagram for the 5-point, adjacent stencil."
\image latex GridOrder4PtLine.pdf "Order diagram for the 5-point, adjacent stencil." width=0.4\textwidth


To ameliorate the problems presented above, we introduce a different
stencil.  Instead of differencing only in the coordinate directions,
we consider differencing in the diagonal directions as well.
The figure below shows a stencil that uses the
four adjacent grid points and a stencil that uses the eight adjacent
and diagonal grid points.  First consider the adjacent stencil.  If
the characteristic comes from the first quadrant, grid point \f$a\f$ would
be used to compute \f$\partial u / \partial x\f$ and grid point \f$b\f$ would
be used to determine \f$\partial u / \partial y\f$.  Next consider the
adjacent-diagonal stencil.  If the characteristic came from the first
sector, grid point \f$a\f$ would be used to compute \f$\partial u / \partial x\f$
and grid point \f$b\f$ would be used to determine
\f$\partial u / \partial x + \partial u / \partial y\f$ from which
\f$\partial u / \partial y\f$ can be determined.  The adjacent-diagonal stencil
increases the number of directions in which information can flow from
four to eight.


\image html 4Pt8PtStencils.jpg "The 5-point, adjacent stencil and the 9-point, adjacent-diagonal stencil."
\image latex 4Pt8PtStencils.pdf "The 5-point, adjacent stencil and the 9-point, adjacent-diagonal stencil." width=0.6\textwidth


Now we consider solving the test problems with the 9-point,
adjacent-diagonal stencil.  The figure below
shows the dependency diagrams.  The arrows identify the \f$\pi/4\f$ sector
from which the characteristic line comes.  (Except for the degenerate
cases where a single grid point is used in the difference scheme.)
Saying that the characteristic direction comes from a sector of angle
\f$\pi/4\f$ is not an accurate description, however, it is more accurate
that saying it comes from one of the four quadrants.  The
adjacent-diagonal stencil reduces the domain of dependence for the
grid points.


\image html GridDependency8PtCircle.jpg "Dependency diagrams for the 9-point, adjacent-diagonal stencil."
\image latex GridDependency8PtCircle.pdf "Dependency diagrams for the 9-point, adjacent-diagonal stencil." width=0.4\textwidth

\image html GridDependency8PtLine.jpg "Dependency diagrams for the 9-point, adjacent-diagonal stencil."
\image latex GridDependency8PtLine.pdf "Dependency diagrams for the 9-point, adjacent-diagonal stencil." width=0.4\textwidth


We turn our attention to developing a correctness criterion for the
adjacent-diagonal stencil.  Let \f$\mu\f$ be the minimum solution among
the labeled grid points.  Each labeled grid point has at least one
known adjacent or diagonal neighbor.  The correct solution at the
other neighbors may be as low as \f$\mu\f$.  We determine the smallest
predicted solution using an unknown neighbor.

First consider the case that an adjacent neighbor is known and a
diagonal neighbor is unknown.  We assign a value of \f$\mu\f$ to the
unknown diagonal neighbor.  If the characteristic line comes from the
sector defined by these two neighbors, the smallest predicted solution
is \f$\mu + \Delta\mathsf{x}\f$.  Next consider the case that the diagonal
neighbor is known and an adjacent neighbor is unknown.  We assign a
value of \f$\mu\f$ to the unknown adjacent neighbor.  The smallest
possible predicted solution in this case is
\f$\mu + \Delta\mathsf{x} / \sqrt{2}\f$.
Thus all labeled grid points with solutions less than or
equal to \f$\mu + \Delta\mathsf{x} / \sqrt{2}\f$ have the correct value.
This turns out to be a useful correctness criterion.
The figures below show the order diagrams for the
first-order, adjacent-diagonal scheme.  We see that at each step, most
of the labeled grid points are determined to be correct.  For the
second test problem, all the labeled vertices are determined to be
correct.  The adjacent-diagonal stencil with this correctness
criterion looks promising.

\image html GridOrder8PtCircle.jpg "Order diagrams for the 9-point, adjacent-diagonal stencil."
\image latex GridOrder8PtCircle.pdf "Order diagrams for the 9-point, adjacent-diagonal stencil." width=0.4\textwidth

\image html GridOrder8PtLine.jpg "Order diagrams for the 9-point, adjacent-diagonal stencil."
\image latex GridOrder8PtLine.pdf "Order diagrams for the 9-point, adjacent-diagonal stencil." width=0.4\textwidth
*/











//=============================================================================
//=============================================================================
/*!
\page hj_adjacentDiagonal Adjacent-Diagonal Difference Schemes


Tsitsiklis presented a first-order accurate
<em>Dial-like algorithm</em>
cite{tsitsiklis:1995}
for solving static Hamilton-Jacobi equations that
uses the diagonal as well as the adjacent neighbors in the finite
differencing.  (In \f$K\f$-dimensional space, a grid point has \f$2 K\f$ adjacent
neighbors and a total of \f$3^K - 1\f$ neighbors.)  Although his algorithm,
which uses all of the neighbors, has the optimal computational complexity
\f$\mathcal{O}(N)\f$, he concluded that it would not be as efficient
as his Dijkstra-like algorithm.  Thus he did not present an implementation.

In this section we will present schemes that use some or all of
the diagonal neighbors in differencing.  We will study both the
accuracy and efficiency of these schemes.

We have seen that the adjacent-diagonal scheme introduced in the
previous section may enable the application of the Marching with a
Correctness Criterion methodology.  In this section we will develop
first and second-order, adjacent-diagonal difference schemes.  Again
we consider solving the eikonal equation, \f$|\nabla u| f = 1\f$ in 2-D.
We know how to difference in the coordinate directions with adjacent
grid points to approximate \f$\partial u / \partial x\f$ and
\f$\partial u / \partial y\f$.  We can also difference in diagonal
directions to get first-order approximations of
\f$\pm \partial u / \partial x \pm \partial u / \partial y\f$.  For example,
\f[
\frac{u_{i,j} - u_{i-1,j-1}}{\Delta x} = \frac{\partial u}{\partial x}
+ \frac{\partial u}{\partial y} + \mathcal{O}\left( \Delta x^2 \right).
\f]
We can also obtain a second-order accurate difference.
\f[
\frac{ 3 u_{i,j} - 4 u_{i-1,j-1} + u_{i-2,j-2} }{ 2 \Delta x } =
\frac{\partial u}{\partial x} + \frac{\partial u}{\partial y}
+ \mathcal{O}\left( \Delta x^3 \right)
\f]
<!--
%\begin{align*}
%  D_{i,j}^{+x+y} u &= \frac{u_{i+1,j+1} - u_{i,j}}{\Delta x} \approx \frac{\partial u}{\partial x} + \frac{\partial u}{\partial y} \\
%  D_{i,j}^{+x-y} u &= \frac{u_{i+1,j-1} - u_{i,j}}{\Delta x} \approx \frac{\partial u}{\partial x} - \frac{\partial u}{\partial y} \\
%  D_{i,j}^{-x+y} u &= \frac{u_{i-1,j+1} - u_{i,j}}{\Delta x} \approx - \frac{\partial u}{\partial x} + \frac{\partial u}{\partial y} \\
%  D_{i,j}^{-x-y} u &= \frac{u_{i-1,j-1} - u_{i,j}}{\Delta x} \approx - \frac{\partial u}{\partial x} - \frac{\partial u}{\partial y}
%\end{align*}
-->
If we know \f$\partial u / \partial x + \partial u / \partial y\f$ from
differencing in a diagonal direction and \f$\partial u / \partial x\f$
from differencing in a horizontal direction, then we can determine
\f$\partial u / \partial y\f$ from the difference of these two.  Thus we
can determine approximations of \f$\partial u / \partial x\f$ and
\f$\partial u / \partial y\f$ from one adjacent difference and one
diagonal difference.

We consider how a grid point that has become known labels its
neighbors using the adjacent-diagonal, first-order, upwind difference
scheme.  For each adjacent neighbor, there are potentially three ways
to compute a new solution there.  In the figure below,
the center grid point has just become known.  The first row
of diagrams show how to label an adjacent neighbor.  Suppose the
adjacent neighbor to the right is not known.  We show three ways the
center grid point can be used to compute the value of this neighbor.
First, only the center grid point is used.  If the grid points that
are adjacent to the center grid point and diagonal to the grid point
being labeled are known, they can be used in the difference scheme as
well.  This accounts for the second two cases.  The second row of
diagrams show how to label a diagonal neighbor.  Again, there are
three ways the center grid point can be used to compute the value of
this neighbor.  First, only the center grid point is used.  If the
grid points that are adjacent to the center grid point and the grid
point being labeled are known, they can be used in the difference
scheme as well.


\image html LabelAdjacentDiagonal.jpg "The three ways of labeling an adjacent neighbor and the three ways of labeling a diagonal neighbor using the adjacent-diagonal, first-order difference scheme."
\image latex LabelAdjacentDiagonal.pdf "The three ways of labeling an adjacent neighbor and the three ways of labeling a diagonal neighbor using the adjacent-diagonal, first-order difference scheme." width=0.9\textwidth


Using an adjacent-diagonal scheme requires us to more closely examine
the upwind concept.  In 1-D, the solution decreases in the
upwind direction.  The characteristic comes from the upwind direction.
In 2-D, any direction in which the solution decreases is an upwind
direction.  If the dot product of a given direction with the characteristic
direction is (positive/negative), then that direction is (downwind/upwind).

For an adjacent difference scheme, the upwind information determines
the quadrant from which the characteristic comes.  The first diagram
in the figure below shows a stencil for an
adjacent difference scheme.  A blue line shows the characteristic
direction.  The upwind coordinate directions are colored red
while the downwind directions are green.  That the characteristic
comes from the third quadrant implies that directions c and d are
upwind.  Conversely, if c and d are upwind directions, then the
characteristic comes from the third quadrant.  For this case, grid
points c and d would be used in the finite difference scheme to
compute the solution at the center grid point.  Choosing the upwind
directions ensures that the CFL condition is satisfied.  That is, the
numerical domain of dependence contains the analytical domain of
dependence.

The situation is different for an adjacent-diagonal scheme.
The second diagram
in the figure below shows a stencil for an
adjacent-diagonal difference scheme.
The characteristic comes from sector 6, but there are four neighboring
grid points which are upwind.  We can use grid points f and g to
determine the center grid point.  If we used either grid points e and f
or grid points g and h, then the scheme would be upwind, but would not
satisfy the CFL condition.  That is, the characteristic direction
would be outside the numerical domain of dependence.  Using sectors
5 or 7 to determine the grid point would lead to an incorrect result.
We will show a simple example of this later in this section.  Although
``CFL satisfying'' would be a more accurate adjective to describe
these adjacent-diagonal schemes than ``upwind,'' we will continue to use the
latter term.


\image html Stencil4Pt8PtUpwind.jpg "The direction of the characteristic is shown in blue.  Upwind directions are shown in red; downwind directions are shown in green."
\image latex Stencil4Pt8PtUpwind.pdf "The direction of the characteristic is shown in blue.  Upwind directions are shown in red; downwind directions are shown in green." width=0.6\textwidth


Below we give the functions that implement the adjacent-diagonal, first-order
difference scheme for the eikonal equation, \f$|\nabla u| f = 1\f$.  The difference
scheme can use a single adjacent grid point (\c differenceAdj()),
a single diagonal grid point (\c differenceDiag()),
or an adjacent and a diagonal grid point (\c differenceAdjDiag()).
The last of these solves a quadratic equation to determine the solution.
If the condition in \c differenceAdjDiag() is not satisfied,
then the characteristic line comes from outside the wedge defined by
the adjacent and diagonal neighbors.  In this case the difference
scheme will not satisfy the CFL condition so we return infinity.


\verbatim
differenceAdj(a):
  return a + dx / f \endverbatim


\verbatim
differenceDiag(a):
  return a + sqrt(2) dx / f \endverbatim


\verbatim
differenceAdjDiag(a, b):
  diff = a - b
  if 0 <= diff <= dx / (sqrt(2) f):
    return adj + sqrt(dx^2 / f^2 - diff^2)
  return Infinity \endverbatim


Now we consider a simple example which demonstrates that the finite
difference scheme must satisfy the CFL condition and not just be
upwind.  We solve the eikonal equation \f$| \nabla u | = 1\f$ on a \f$3
\times 3\f$ grid.  The grid spacing is unity and initially the lower
left grid point is set to zero.  The solution is the Euclidean
distance from that grid point.
The figure below shows the initial condition
and the analytical solution on the grid.

<!--CONTINUE
\image html .jpg ""
\image latex .pdf "" width=\textwidth

\begin{figure}[tbh!]
  \begin{center}
    \small
    \setlength{\unitlength}{0.08\textwidth}

    %%
    %% Initial Condition
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\textcolor{red}{\f$\infty\f$}}
      \put(2,0){\textcolor{red}{\f$\infty\f$}}
      \put(0,1){\textcolor{red}{\f$\infty\f$}}
      \put(1,1){\textcolor{red}{\f$\infty\f$}}
      \put(2,1){\textcolor{red}{\f$\infty\f$}}
      \put(0,2){\textcolor{red}{\f$\infty\f$}}
      \put(1,2){\textcolor{red}{\f$\infty\f$}}
      \put(2,2){\textcolor{red}{\f$\infty\f$}}

      \put(0.1,-0.5){Initial Condition}
    \end{picture}
    %%
    %% Analytical Solution
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$\sqrt{2}\f$}
      \put(2,1){\f$\sqrt{5}\f$}
      \put(0,2){\f$2\f$}
      \put(1,2){\f$\sqrt{5}\f$}
      \put(2,2){\f$2 \sqrt{2}\f$}

      \put(0.1,-0.5){Analytical Solution}
    \end{picture}

  \end{center}
  \caption{The initial condition and the analytic solution on the grid.}
  \label{figure initial solution CFL}
\end{figure}
-->


The figure below shows the result of using the
Fast Marching Method with the first-order, adjacent-diagonal scheme.
The scheme is upwind, but we do not enforce the CFL condition to
restrict which differences are applied.  Each of the labeling operations
are depicted.  In step 1, grid point \f$(0,0)\f$ labels its adjacent and
diagonal neighbors.  We see the first sign of trouble in steps 2 and 3.
In these steps, grid points \f$(0,0)\f$ and \f$(1,0)\f$ are used to label \f$(1,1)\f$
with the value \f$1\f$.  The values at these two known grid points indicate
that the characteristic is in the direction of the positive \f$x\f$ axis.
Thus the characteristic does not come from the sector described by
the three grid points.  As a result, the predicted solution is erroneous.
This problem reoccurs in the subsequent steps.  It is apparent that
the solution will not converge as the grid is refined.


<!--CONTINUE
\image html .jpg ""
\image latex .pdf "" width=\textwidth

\begin{figure}[tbp!]
  \begin{center}
    \small
    \setlength{\unitlength}{0.08\textwidth}

    %%
    %% Step 1
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\textcolor{green}{\f$1\f$}}
      \put(2,0){\textcolor{red}{\f$\infty\f$}}
      \put(0,1){\textcolor{green}{\f$1\f$}}
      \put(1,1){\textcolor{green}{\f$\sqrt{2}\f$}}
      \put(2,1){\textcolor{red}{\f$\infty\f$}}
      \put(0,2){\textcolor{red}{\f$\infty\f$}}
      \put(1,2){\textcolor{red}{\f$\infty\f$}}
      \put(2,2){\textcolor{red}{\f$\infty\f$}}

      %%\drawline(0.2,0.1)(1,0.1)
      \put(0.3,0.1){\vector(1,0){0.6}}
      \put(0.1,0.3){\vector(0,1){0.6}}
      \put(0.3,0.3){\vector(1,1){0.6}}

      \put(0.7,-0.5){Step 1}
    \end{picture}
    %%
    %% Steps 2 and 3
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\textcolor{green}{\f$2\f$}}
      \put(0,1){\f$1\f$}
      \put(1,1){\textcolor{green}{\f$1\f$}}
      \put(2,1){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}
      \put(0,2){\textcolor{green}{\f$2\f$}}
      \put(1,2){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}
      \put(2,2){\textcolor{red}{\f$\infty\f$}}

      %% label from (1,0)
      \put(1.3,0.1){\vector(1,0){0.6}}
      \put(1.1,0.3){\vector(0,1){0.6}}
      \put(1.3,0.3){\vector(1,1){0.6}}

      \put(0.65,0.3){\vector(1,2){0.3}}
      \drawline(0.3,0.25)(0.65,0.3)
      \drawline(0.9,0.25)(0.65,0.3)

      %% label from (0,1)
      \put(0.3,1.1){\vector(1,0){0.6}}
      \put(0.1,1.3){\vector(0,1){0.6}}
      \put(0.3,1.3){\vector(1,1){0.6}}

      \put(0.3,0.65){\vector(2,1){0.6}}
      \drawline(0.25,0.3)(0.3,0.65)
      \drawline(0.25,0.9)(0.3,0.65)

      \put(0.2,-0.5){Steps 2 and 3}
    \end{picture}
    %%
    %% Step 4
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\textcolor{green}{\f$2\f$}}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$1\f$}
      \put(2,1){\textcolor{green}{\f$2\f$}}
      \put(0,2){\textcolor{green}{\f$2\f$}}
      \put(1,2){\textcolor{green}{\f$2\f$}}
      \put(2,2){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}

      %% label from (1,1)
      \put(1.3,1.1){\vector(1,0){0.6}}
      \put(1.1,1.3){\vector(0,1){0.6}}

      \put(1.3,0.9){\vector(1,-1){0.6}}
      \put(1.3,1.3){\vector(1,1){0.6}}
      \put(0.9,1.3){\vector(-1,1){0.6}}

      \put(0.65,1.3){\vector(1,2){0.3}}
      \drawline(0.3,1.25)(0.65,1.3)
      \drawline(0.9,1.25)(0.65,1.3)

      \put(0.55,1.3){\vector(-1,2){0.3}}
      \drawline(0.3,1.25)(0.55,1.3)
      \drawline(0.9,1.25)(0.55,1.3)

      \put(1.3,0.65){\vector(2,1){0.6}}
      \drawline(1.25,0.3)(1.3,0.65)
      \drawline(1.25,0.9)(1.3,0.65)

      \put(1.3,0.55){\vector(2,-1){0.6}}
      \drawline(1.25,0.3)(1.3,0.55)
      \drawline(1.25,0.9)(1.3,0.55)

      \put(0.7,-0.5){Step 4}
    \end{picture}

    %%
    %% Steps 5 and 6
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$1\f$}
      \put(2,1){\textcolor{green}{\f$2\f$}}
      \put(0,2){\f$2\f$}
      \put(1,2){\textcolor{green}{\f$2\f$}}
      \put(2,2){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}

      %% label from (2,0)
      \put(2.1,0.3){\vector(0,1){0.6}}

      \put(1.65,0.3){\vector(1,2){0.3}}
      \drawline(1.3,0.25)(1.65,0.3)
      \drawline(1.9,0.25)(1.65,0.3)

      %% label from (0,2)
      \put(0.3,2.1){\vector(1,0){0.6}}
      \put(0.3,1.65){\vector(2,1){0.6}}
      \drawline(0.25,1.3)(0.3,1.65)
      \drawline(0.25,1.9)(0.3,1.65)

      \put(0.2,-0.5){Steps 5 and 6}
    \end{picture}
    %%
    %% Steps 7 and 8
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$1\f$}
      \put(2,1){\f$2\f$}
      \put(0,2){\f$2\f$}
      \put(1,2){\f$2\f$}
      \put(2,2){\textcolor{green}{\f$2\f$}}

      %% label from (2,1)
      \put(2.1,1.3){\vector(0,1){0.6}}

      \put(1.65,1.3){\vector(1,2){0.3}}
      \drawline(1.3,1.25)(1.65,1.3)
      \drawline(1.9,1.25)(1.65,1.3)

      %% label from (1,2)
      \put(1.3,2.1){\vector(1,0){0.6}}
      \put(1.3,1.65){\vector(2,1){0.6}}
      \drawline(1.25,1.3)(1.3,1.65)
      \drawline(1.25,1.9)(1.3,1.65)

      \put(0.2,-0.5){Steps 7 and 8}
    \end{picture}
    %%
    %% Solution
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$1\f$}
      \put(2,1){\f$2\f$}
      \put(0,2){\f$2\f$}
      \put(1,2){\f$2\f$}
      \put(2,2){\f$2\f$}

      \put(0.5,-0.5){Solution}
    \end{picture}

  \end{center}
  \caption{An adjacent-diagonal difference scheme that is upwind, but does
    not satisfy the CFL condition.}
  \label{figure upwind violate CFL}
\end{figure}
-->


By contrast, the figure below shows the result of using
the first-order, adjacent-diagonal scheme that satisfies the CFL condition.
Each of the labeling operations which satisfy the CFL condition
are depicted.  Note that fewer of these operations are permissible.
This approach results in a convergent scheme.



<!--CONTINUE
\image html .jpg ""
\image latex .pdf "" width=\textwidth

\begin{figure}[tbp!]
  \begin{center}
    \small
    \setlength{\unitlength}{0.08\textwidth}

    %%
    %% Step 1
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\textcolor{green}{\f$1\f$}}
      \put(2,0){\textcolor{red}{\f$\infty\f$}}
      \put(0,1){\textcolor{green}{\f$1\f$}}
      \put(1,1){\textcolor{green}{\f$\sqrt{2}\f$}}
      \put(2,1){\textcolor{red}{\f$\infty\f$}}
      \put(0,2){\textcolor{red}{\f$\infty\f$}}
      \put(1,2){\textcolor{red}{\f$\infty\f$}}
      \put(2,2){\textcolor{red}{\f$\infty\f$}}

      %%\drawline(0.2,0.1)(1,0.1)
      \put(0.3,0.1){\vector(1,0){0.6}}
      \put(0.1,0.3){\vector(0,1){0.6}}
      \put(0.3,0.3){\vector(1,1){0.6}}

      \put(0.7,-0.5){Step 1}
    \end{picture}
    %%
    %% Steps 2 and 3
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\textcolor{green}{\f$2\f$}}
      \put(0,1){\f$1\f$}
      \put(1,1){\textcolor{green}{\f$\sqrt{2}\f$}}
      \put(2,1){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}
      \put(0,2){\textcolor{green}{\f$2\f$}}
      \put(1,2){\textcolor{green}{\f$1 \!\! + \!\! \sqrt{2}\f$}}
      \put(2,2){\textcolor{red}{\f$\infty\f$}}

      %% label from (1,0)
      \put(1.3,0.1){\vector(1,0){0.6}}
      \put(1.1,0.3){\vector(0,1){0.6}}
      \put(1.3,0.3){\vector(1,1){0.6}}

      %% label from (0,1)
      \put(0.3,1.1){\vector(1,0){0.6}}
      \put(0.1,1.3){\vector(0,1){0.6}}
      \put(0.3,1.3){\vector(1,1){0.6}}

      \put(0.1,-0.5){Steps 2 and 3}
    \end{picture}
    %%
    %% Step 4
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\textcolor{green}{\f$2\f$}}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$\sqrt{2}\f$}
      \put(2,1.2){\textcolor{green}{\f$\sqrt{2} +\f$}}
      \put(2,0.8){\textcolor{green}{\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}}
      \put(0,2){\textcolor{green}{\f$2\f$}}
      \put(0.5,2.4){\textcolor{green}{\f$\sqrt{2} +\f$}}
      \put(0.5,2){\textcolor{green}{\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}}
      \put(2,2){\textcolor{green}{\f$2 \sqrt{2}\f$}}

      %% label from (1,1)
      \put(1.45,1.1){\vector(1,0){0.45}}
      \put(1.1,1.3){\vector(0,1){0.6}}

      \put(1.3,0.9){\vector(1,-1){0.6}}
      \put(1.3,1.3){\vector(1,1){0.6}}
      \put(0.9,1.3){\vector(-1,1){0.6}}

      \put(0.65,1.3){\vector(1,2){0.3}}
      \drawline(0.3,1.25)(0.65,1.3)
      \drawline(0.9,1.25)(0.65,1.3)

      \put(1.3,0.65){\vector(2,1){0.6}}
      \drawline(1.25,0.3)(1.3,0.65)
      \drawline(1.25,0.9)(1.3,0.65)

      \put(0.7,-0.5){Step 4}
    \end{picture}

    %%
    %% Steps 5 and 6
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$\sqrt{2}\f$}
      \put(2,1.2){\textcolor{green}{\f$\sqrt{2} +\f$}}
      \put(2,0.8){\textcolor{green}{\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}}
      \put(0,2){\f$2\f$}
      \put(0.5,2.4){\textcolor{green}{\f$\sqrt{2} +\f$}}
      \put(0.5,2){\textcolor{green}{\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}}
      \put(2,2){\textcolor{green}{\f$2 \sqrt{2}\f$}}

      %% label from (2,0)
      \put(2.1,0.3){\vector(0,1){0.4}}

      %% label from (0,2)
      \put(0.2,2.1){\vector(1,0){0.3}}

      \put(0.2,-0.5){Steps 5 and 6}
    \end{picture}
    %%
    %% Steps 7 and 8
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$\sqrt{2}\f$}
      \put(2,1.2){\f$\sqrt{2} +\f$}
      \put(2,0.8){\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}
      \put(0,2){\f$2\f$}
      \put(0.5,2.4){\f$\sqrt{2} +\f$}
      \put(0.5,2){\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}
      \put(2,2){\textcolor{green}{\f$2 \sqrt{2}\f$}}

      %% label from (2,1)
      \put(2.1,1.3){\vector(0,1){0.6}}

      %% label from (1,2)
      \put(1.7,2.1){\vector(1,0){0.3}}

      \put(0.1,-0.5){Steps 7 and 8}
    \end{picture}
    %%
    %% Solution
    %%
    \begin{picture}(4,3.5)(0,-1)
      \put(0,0){\f$0\f$}
      \put(1,0){\f$1\f$}
      \put(2,0){\f$2\f$}
      \put(0,1){\f$1\f$}
      \put(1,1){\f$\sqrt{2}\f$}
      \put(2,1.2){\f$\sqrt{2} +\f$}
      \put(2,0.8){\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}
      \put(0,2){\f$2\f$}
      \put(0.5,2.4){\f$\sqrt{2} +\f$}
      \put(0.5,2){\f$\sqrt{\! 2 \sqrt{2} \!\! - \!\! 2}\f$}
      \put(2,2){\f$2 \sqrt{2}\f$}

      \put(0.5,-0.5){Solution}
    \end{picture}

  \end{center}
  \caption{An adjacent-diagonal difference scheme that satisfies the CFL
    condition.}
  \label{figure satisfy CFL}
\end{figure}
-->
*/










//=============================================================================
//=============================================================================
/*!
\page hj_computationalComplexity Computational Complexity

Now we determine the computational complexity of the MCC algorithm for
solving the eikonal equation \f$| \nabla u | f = 1\f$.  Let the values of \f$f\f$
be in the interval \f$[A \ldots B]\f$ and let \f$R = B / A\f$.  Consider the
MCC algorithm in progress.  Let \f$\mu\f$ be the minimum solution of the
labeled grid points.  The predicted solutions at the labeled grid
points are in the range \f$[\mu \ldots \mu + \sqrt{2} \Delta x / A )\f$.  When one
applies the correctness criterion, at least all of the labeled
vertices with distances less than or equal to \f$\mu + \frac{\Delta x}{\sqrt{2} B}\f$
will become known.  Thus at the next step, the minimum
labeled solution will be at least \f$\mu + \frac{\Delta x}{\sqrt{2} B}\f$.  At
each step of the algorithm, the minimum labeled solution increases by
at least \f$\frac{\Delta x}{\sqrt{2} B}\f$.  This means that a grid point may
be in the labeled set for at most
\f$\frac{\sqrt{2} \Delta x}{A} / \frac{\Delta x}{\sqrt{2} B} = 2 B / A = 2 R\f$
steps.  The computational cost of
applying the correctness criteria is thus \f$\mathcal{O}(R N)\f$.  The
cost of labeling is \f$\mathcal{O}(N)\f$.  Since a grid point is simply
added to the end of a list or array when it becomes labeled, the cost
of adding and removing labeled grid points is \f$\mathcal{O}(N)\f$.  Thus
the computation complexity of the MCC algorithm is \f$\mathcal{O}(R N)\f$.
*/














//=============================================================================
//=============================================================================
/*!
\page hj_performance Performance Comparison of the Finite Difference Schemes with the FMM


<!---------------------------------------------------------------------------->
\section hj_performance_test Test Problems


We consider three test problems.  In each problem, the distance is computed
from a point or set of points.  We consider cases in which the solution is
smooth, the solution has high curvature and the solution has shocks, i.e.,
the solution is not everywhere differentiable.
The figures below
show the test problem with a smooth solution.  The grid spans the
domain \f$[-1/2 .. 1/2] \times [-1/2 .. 1/2]\f$.  The distance is computed
from a single point at \f$(-3/4, -3/4)\f$.


\image html InitialConditionOutside.jpg "Test problem for a smooth solution.  The top diagram depicts a 10 by 10 grid.  The distance is computed from the point outside the grid depicted as a solid black disk. The red grid points show where the initial condition for first-order schemes is specified.  The green grid points show the additional grid points where the initial condition is specified for second-order schemes."
\image latex InitialConditionOutside.pdf "Test problem for a smooth solution.  The top diagram depicts a 10 by 10 grid.  The distance is computed from the point outside the grid depicted as a solid black disk. The red grid points show where the initial condition for first-order schemes is specified.  The green grid points show the additional grid points where the initial condition is specified for second-order schemes." width=0.5\textwidth

\image html ExactOutside.jpg "Plot of the smooth solution."
\image latex ExactOutside.pdf "Plot of the smooth solution." width=0.5\textwidth


Next we consider
the test problem with high curvature.  The distance is computed
from a single point in the center of the grid.  The solution has high
curvature near the center.  This is where the difference schemes will
make the largest errors.


\image html InitialConditionCenter.jpg "Test problem for a solution with high curvature. The diagram shows a 10 by 10 grid.  The distance is computed from a point at the center of the grid. The red grid points show where the initial condition is specified for first-order and second-order schemes."
\image latex InitialConditionCenter.pdf "Test problem for a solution with high curvature. The diagram shows a 10 by 10 grid.  The distance is computed from a point at the center of the grid. The red grid points show where the initial condition is specified for first-order and second-order schemes." width=0.5\textwidth

\image html ExactCenter.jpg "Plot of the solution."
\image latex ExactCenter.pdf "Plot of the solution." width=0.5\textwidth


Finally we show the problem in which the solution has shocks.  The grid spans
the domain \f$[-1/2 .. 1/2] \times [-1/2 .. 1/2]\f$.  The distance is
computed from \f$16\f$ points on the circle of unit radius, centered at
the origin.  There are shocks along lines that are equidistant from
two points.  This test case produces shock lines at a variety of
angles.

\image html InitialConditionCircleOutside.jpg "Test problem for a solution with shocks. The diagram depicts a 10 by 10 grid.  The distance is computed from the points on the unit circle.  Lines show the locations of the shocks. The red grid points show where the initial condition for first-order schemes is specified.  The green grid points show the additional grid points where the initial condition is specified for second-order schemes."
\image latex InitialConditionCircleOutside.pdf "Test problem for a solution with shocks. The diagram depicts a 10 by 10 grid.  The distance is computed from the points on the unit circle.  Lines show the locations of the shocks. The red grid points show where the initial condition for first-order schemes is specified.  The green grid points show the additional grid points where the initial condition is specified for second-order schemes." width=0.7\textwidth

\image html ExactCircleOutside.jpg "Plot of the solution."
\image latex ExactCircleOutside.pdf "Plot of the solution." width=0.5\textwidth





<!---------------------------------------------------------------------------->
\section hj_performance_convergence Convergence

<!---------------------------------------------------------------------------->
\subsection hj_performance_convergence_smooth Smooth Solution

First we examine the behavior of the finite difference schemes on the
test problem with a smooth solution.  We use the schemes to solve the
problem on a \f$40 \times 40\f$ grid.  The figures below
show plots of the error for the four difference schemes.  For the
adjacent stencils, the largest errors are in the diagonal direction.
This reflects that the differencing is done in the coordinate
directions.  The first-order, adjacent scheme has a large error.  For
the second-order scheme, the error is much smaller.  For the
adjacent-diagonal stencils, the largest errors are in the directions
which lie between the coordinate and diagonal directions.  This is
expected as the schemes difference in the coordinate and diagonal
directions.  The first-order, adjacent-diagonal scheme has
significantly smaller errors than the first-order, adjacent scheme.
The second-order, adjacent-diagonal scheme has the smallest errors.


\image html ErrorOutside_4_1.jpg "Plot of the error for a smooth solution.  First-order, adjacent scheme."
\image latex ErrorOutside_4_1.pdf "Plot of the error for a smooth solution.  First-order, adjacent scheme." width=0.5\textwidth

\image html ErrorOutside_4_2.jpg "Plot of the error for a smooth solution.  Second-order, adjacent scheme."
\image latex ErrorOutside_4_2.pdf "Plot of the error for a smooth solution.  Second-order, adjacent scheme." width=0.5\textwidth

\image html ErrorOutside_8_1.jpg "Plot of the error for a smooth solution.  First-order, adjacent-diagonal scheme."
\image latex ErrorOutside_8_1.pdf "Plot of the error for a smooth solution.  First-order, adjacent-diagonal scheme." width=0.5\textwidth

\image html ErrorOutside_8_2.jpg "Plot of the error for a smooth solution.  Second-order, adjacent-diagonal scheme."
\image latex ErrorOutside_8_2.pdf "Plot of the error for a smooth solution.  Second-order, adjacent-diagonal scheme." width=0.5\textwidth


Now we examine the convergence of the solution using the difference
schemes.  The first graph below
shows the \f$L_1\f$ error versus the grid spacing for grids ranging in
size from \f$10 \times 10\f$ to \f$5120 \times 5120\f$.  We see that going
from an adjacent stencil to an adjacent-diagonal stencil reduces the
error by about a factor of 10.  The second-order schemes have a higher
rate of convergence than the first-order schemes.  The \f$L_\infty\f$
error shows the same behavior.

\image html ErrorOutside2DL1.jpg "L_1 error for a smooth solution. Log-log plot of the error versus the grid spacing."
\image latex ErrorOutside2DL1.pdf "L_1 error for a smooth solution. Log-log plot of the error versus the grid spacing." width=\textwidth

\image html ErrorOutside2DLinf.jpg "Maximum error for a smooth solution. Log-log plot of the error versus the grid spacing."
\image latex ErrorOutside2DLinf.pdf "Maximum error for a smooth solution. Log-log plot of the error versus the grid spacing." width=\textwidth



The table below shows the numerical rate of
convergence for the difference schemes.  (If the the error is
proportional to \f$\Delta x^\alpha\f$, where \f$\Delta x\f$ is the grid
spacing, then \f$\alpha\f$ is the rate of convergence.)  We see that for
the smooth solution, the first-order schemes have first-order
convergence and the second-order schemes have second-order
convergence.


<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.998 <td> 0.992
<tr>
<td> Second-Order, Adjacent <td> 1.995 <td> 1.995
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.995 <td> 0.997
<tr>
<td> Second-Order, Adjacent-Diagonal <td> 1.995 <td> 1.997
</table>




<!---------------------------------------------------------------------------->
\subsection hj_performance_convergence_high Solution with High Curvature

Next we examine the behavior of the finite difference schemes on the
test problem with high curvature.  Again we use the schemes to solve
the problem on a \f$40 \times 40\f$ grid.  The figures below
show plots of the error for the four difference schemes.  The
first-order, adjacent scheme has significant errors in the region of
high curvature, but accumulates larger errors in the low curvature
region (especially in diagonal directions).  Going to a second-order
scheme introduces larger errors in the high curvature region.  However
the second-order, adjacent scheme is more accurate where the solution
has low curvature.  The first-order, adjacent-diagonal scheme is
better at handling the high curvature region.  Like the first-order,
adjacent scheme, the error noticeably accumulates in the low curvature
region, but to a lesser extent.  The second-order, adjacent-diagonal
scheme has relatively large errors in the high curvature region, but
is very accurate elsewhere.


\image html ErrorCenter_4_1.jpg "Plot of the error for a solution with high curvature. First-order, adjacent scheme."
\image latex ErrorCenter_4_1.pdf "Plot of the error for a solution with high curvature. First-order, adjacent scheme." width=0.5\textwidth

\image html ErrorCenter_4_2.jpg "Plot of the error for a solution with high curvature. Second-order, adjacent scheme."
\image latex ErrorCenter_4_2.pdf "Plot of the error for a solution with high curvature. Second-order, adjacent scheme." width=0.5\textwidth

\image html ErrorCenter_8_1.jpg "Plot of the error for a solution with high curvature. First-order, adjacent-diagonal scheme."
\image latex ErrorCenter_8_1.pdf "Plot of the error for a solution with high curvature. First-order, adjacent-diagonal scheme." width=0.5\textwidth

\image html ErrorCenter_8_2.jpg "Plot of the error for a solution with high curvature. Second-order, adjacent-diagonal scheme."
\image latex ErrorCenter_8_2.pdf "Plot of the error for a solution with high curvature. Second-order, adjacent-diagonal scheme." width=0.5\textwidth


We examine the convergence of the solution with high curvature using
the difference schemes.  The first graph below shows
the \f$L_1\f$ error versus the grid spacing for grids ranging in size from
\f$10 \times 10\f$ to \f$5120 \times 5120\f$.  First we note that the second
order schemes have only a slightly higher rate of convergence than the
first-order methods.  Using an adjacent-diagonal stencil still
significantly reduces the error.  The \f$L_\infty\f$ error shows the same behavior.


\image html ErrorCenter2DL1.jpg "L_1 error for a solution with high curvature. Log-log plot of the error versus the grid spacing."
\image latex ErrorCenter2DL1.pdf "L_1 error for a solution with high curvature. Log-log plot of the error versus the grid spacing." width=\textwidth

\image html ErrorCenter2DLinf.jpg "Maximum error for a solution with high curvature. Log-log plot of the error versus the grid spacing."
\image latex ErrorCenter2DLinf.pdf "Maximum error for a solution with high curvature. Log-log plot of the error versus the grid spacing." width=\textwidth


The table below shows the numerical rate of convergence
for the difference schemes.  For the solution with high curvature,
the first-order schemes have less than first-order convergence.  For
both the adjacent and the adjacent-diagonal schemes it is about \f$0.85\f$.
The second-order schemes have first-order convergence.  This is because they
make first-order errors in the region of high curvature and then propagate
this error through the region of low curvature where they are second-order
accurate.

<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.840 <td> 0.848
<tr>
<td> Second-Order, Adjacent <td> 1.002 <td> 1.000
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.853 <td> 0.855
<tr>
<td> Second-Order, Adjacent-Diagonal <td> 0.998 <td> 0.999
</table>





<!---------------------------------------------------------------------------->
\subsection hj_performance_convergence_shocks Solution with Shocks


Finally, we examine the behavior of the finite difference schemes on the test
problem with shocks.  The problem is solved on a \f$40 \times 40\f$ grid.
The figures below show plots of the error for
the four difference schemes.  For the first-order, adjacent scheme,
the errors in the smooth regions and along the shocks have
approximately the same magnitude.  Away from the shocks, the second
order, adjacent scheme has low errors, but there are relatively large
errors in a wide band around the shocks.  The first-order,
adjacent-diagonal scheme has significantly smaller errors than the
corresponding adjacent scheme.  Like the adjacent scheme, the errors
in the smooth regions and near the shocks have approximately the same
magnitude.  The second-order, adjacent-diagonal scheme is very
accurate in the smooth regions.  Like the second-order, adjacent
scheme, it has relatively large errors near the shocks, but these
large errors are confined to narrow bands around the shocks.


\image html ErrorCircleOutside_4_1.jpg "Plot of the error for a solution with shocks.  First-order, adjacent scheme."
\image latex ErrorCircleOutside_4_1.pdf "Plot of the error for a solution with shocks.  First-order, adjacent scheme." width=0.5\textwidth

\image html ErrorCircleOutside_4_2.jpg "Plot of the error for a solution with shocks.  Second-order, adjacent scheme."
\image latex ErrorCircleOutside_4_2.pdf "Plot of the error for a solution with shocks.  Second-order, adjacent scheme." width=0.5\textwidth

\image html ErrorCircleOutside_8_1.jpg "Plot of the error for a solution with shocks.  First-order, adjacent-diagonal scheme."
\image latex ErrorCircleOutside_8_1.pdf "Plot of the error for a solution with shocks.  First-order, adjacent-diagonal scheme." width=0.5\textwidth

\image html ErrorCircleOutside_8_2.jpg "Plot of the error for a solution with shocks.  Second-order, adjacent-diagonal scheme."
\image latex ErrorCircleOutside_8_2.pdf "Plot of the error for a solution with shocks.  Second-order, adjacent-diagonal scheme." width=0.5\textwidth


We examine the convergence of the solution with shocks using the four
difference schemes.  The first graph below
shows the \f$L_1\f$ error versus the grid spacing for grids ranging in
size from \f$10 \times 10\f$ to \f$5120 \times 5120\f$.  As before, using an
adjacent-diagonal stencil significantly reduces the error.  For small
grids, the first-order and second-order schemes have about the same rate of
convergence.  For larger grids, the second-order schemes have a higher
rate of convergence.  Unlike the other tests,
each of the schemes has about the same rate of convergence for
the \f$L_\infty\f$ error.


\image html ErrorCircleOutside2DL1.jpg "Log-log plot of the L_1 error versus the grid spacing for the solution with shocks."
\image latex ErrorCircleOutside2DL1.pdf "Log-log plot of the L_1 error versus the grid spacing for the solution with shocks." width=\textwidth

\image html ErrorCircleOutside2DLinf.jpg "Log-log plot of the maximum error versus the grid spacing for the solution with shocks."
\image latex ErrorCircleOutside2DLinf.pdf "Log-log plot of the maximum error versus the grid spacing for the solution with shocks." width=\textwidth


The table below shows the numerical rate of
convergence for the solution with shocks using the four difference
schemes.  For the \f$L_1\f$ error, the schemes have about the same rate of
convergence as they do on the smooth solution.  This is because most
of the grid points are in the smooth region of the solution.  Next we
consider the \f$L_\infty\f$ error.  The first-order, adjacent scheme and
both of the adjacent-diagonal schemes have first-order convergence.
The second-order, adjacent scheme has less than first-order
convergence.  This is because the wide stencil introduces first-order
errors from the shock region into the smooth solution region in a band
around the shock.  This causes errors which are larger than first
order to accumulate in a band around the shock.  The second-order,
adjacent-diagonal scheme did not have this problem for this test.  It
confined the first-order errors to narrow bands around the shocks.


<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.977 <td> 1.001
<tr>
<td> Second-Order, Adjacent <td> 1.984 <td> 0.915
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.990 <td> 0.986
<tr>
<td> Second-Order, Adjacent-Diagonal <td> 1.992 <td> 0.994
</table>





<!---------------------------------------------------------------------------->
\section hj_performance_execution Execution Time

Below we show the
execution times for the four difference schemes using the Fast
Marching Method.  The distance from a center point is computed on
grids ranging in size from \f$10 \times 10\f$ to \f$5120 \times 5120\f$.
Using the adjacent-diagonal stencils is more computationally expensive
than using the adjacent stencils.  For a small grid, using the first
order, adjacent-diagonal scheme increases the execution time by about
\f$25 \%\f$ over using the adjacent scheme.  Using the second-order,
adjacent-diagonal scheme increases the execution time by about \f$50 \%\f$
over the adjacent scheme.  This margin decreases as the grid size
increases.  The execution time per grid point increases as the grid
grows.  However, the performance difference between each of the
methods remains roughly constant.  This reflects the fact that the
cost per grid point of labeling does not depend on the grid size,
but the cost per grid point of
maintaining the binary heap increases with the increasing grid size.

\image html ExecutionTime2DFM.jpg "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching method with different stencils."
\image latex ExecutionTime2DFM.pdf "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching method with different stencils." width=\textwidth
*/









//=============================================================================
//=============================================================================
/*!
\page hj_performanceComparison Performance Comparison of the FMM and the MCC Algorithm


<!---------------------------------------------------------------------------->
\section hj_performanceComparison_memory Memory Usage

Because the Marching with a Correctness Criterion algorithm has a simpler
data structure for storing labeled grid points, it requires a little
less memory than the Fast Marching Method.  First consider the MCC
algorithm.  It has two arrays of size \f$N\f$ where \f$N\f$ is the number of
grid points.  There is an array of floating point numbers for the
solution and an an array to store the status of the grid points.  It
also has a variable sized array of pointers to store the labeled grid
points.  Since the number of labeled grid points is typically much
smaller than the total number of grid points, the memory required for
the labeled array is negligible compared to the solution array and
status array.  The FMM has these three arrays as well.  It uses
the labeled array as a binary heap.  In addition, the FMM
requires a size \f$N\f$ array of pointers into the heap.  This is used to
adjust the position of a labeled grid point in the heap when the
solution decreases.  Thus the MCC algorithm has two size \f$N\f$ arrays
while the FMM has three.  Suppose that one uses single precision
floating point numbers for the solution and integers for the status.
Single precision floating point numbers, integers and pointers
typically each have a size of 4 words.  Thus the FMM requires
about \f$1/2\f$ more memory than the MCC algorithm.  For
double precision floating point numbers (8 words) the FMM
requires a third more memory.


<!---------------------------------------------------------------------------->
\section hj_performanceComparison_execution Execution Time


Now we compare the execution times of the MCC algorithm and the FMM
using the first-order and second-order, adjacent-diagonal schemes.
We also implement a method that measures the execution time of an
ideal algorithm for solving static Hamilton-Jacobi equations with an
ordered, upwind scheme.  For this ideal algorithm, the labeled grid
points are stored in a first-in-first-out queue.  (This was
implemented with the deque data structure in the C++ standard template
library cite{austern:1999}.)  At each step the labeled grid point at
the front of the queue becomes known and it labels its neighbors.  The
algorithm performs a breadth-first traversal of the grid points as the
solution is marched out.  This approach does not produce the correct
solution.  It represents the ideal execution time because it
determines which labeled grid point is ``correct'' in small constant time.

In the figure below we show the execution times
for the first-order, adjacent-diagonal scheme.   The graph shows the linear
computational complexity of the MCC algorithm.  Its performance comes close
to that of the ideal algorithm.  We can see the \f$N \log N\f$ complexity of
the FMM, but it still performs well.  For the largest grid size, its
execution time is about twice that of the MCC algorithm.

\image html ExecutionTime2D8Pt1st.jpg "Log-linear plot of the execution time per grid point versus the number of grid points for the Fast Marching Method, the Marching with a Correctness Criterion algorithm and the ideal algorithm using a first-order, adjacent-diagonal scheme."
\image latex ExecutionTime2D8Pt1st.pdf "Log-linear plot of the execution time per grid point versus the number of grid points for the Fast Marching Method, the Marching with a Correctness Criterion algorithm and the ideal algorithm using a first-order, adjacent-diagonal scheme." width=0.75\textwidth


Next we show the execution times
for the second-order, adjacent-diagonal scheme.   The graph has the same
features as that for the first-order scheme.  However, the finite difference
operations are more expensive so the relative differences in performance are
smaller.

\image html ExecutionTime2D8Pt2nd.jpg "Log-linear plot of the execution time per grid point versus the number of grid points for the Fast Marching Method, the Marching with a Correctness Criterion algorithm and the ideal algorithm using a second-order, adjacent-diagonal scheme."
\image latex ExecutionTime2D8Pt2nd.pdf "Log-linear plot of the execution time per grid point versus the number of grid points for the Fast Marching Method, the Marching with a Correctness Criterion algorithm and the ideal algorithm using a second-order, adjacent-diagonal scheme." width=0.75\textwidth


Recall that the computational complexity of the MCC algorithm
for solving the eikonal equation \f$| \nabla u | f = 1\f$ is
\f$\mathcal{O}(R N)\f$ where \f$R\f$ is the ratio of the maximum to minumum
propagation speed \f$f\f$.  We consider the effect of \f$R\f$ on the
execution times of the marching methods.  We choose a speed function
\f$f\f$ that varies between \f$1\f$ and \f$R\f$ on the domain \f$[0 .. 1]^2\f$:
\f[
f(x,y) = 1 + \frac{R-1}{2}(1 + \sin(6 \pi (x + y)) ).
\f]
We solve the eikonal equation on a \f$1000 \times 1000\f$ grid with the boundary
condition \f$u(1/2, 1/2) = 0\f$ as we vary \f$R\f$ from \f$1\f$ to \f$1024\f$.
The first figure below shows the execution times
for the first-order, adjacent-diagonal scheme.
The next shows results
for the second-order, adjacent-diagonal scheme.
We see that varying \f$R\f$
has little effect on the performance of the FMM.  It has a moderate effect
on the performance of the MCC algorithm.  As expected, the execution time
increases with increasing \f$R\f$.  However, the increase is modest because
a correctness test is an inexpensive operation compared to
a labeling operation using the finite difference scheme.  The MCC algorithm
out-performs the FMM for all tested values of \f$R\f$.


\image html ExecutionTimeRatio2D1st.jpg "Log-log plot of the execution time versus the ratio of the maximum to minimum propagation speed in the eikonal equation. We compare the Fast Marching Method and the Marching with a Correctness Criterion algorithm using a first-order, adjacent-diagonal scheme."
\image latex ExecutionTimeRatio2D1st.pdf "Log-log plot of the execution time versus the ratio of the maximum to minimum propagation speed in the eikonal equation. We compare the Fast Marching Method and the Marching with a Correctness Criterion algorithm using a first-order, adjacent-diagonal scheme." width=0.65\textwidth


\image html ExecutionTimeRatio2D2nd.jpg "Log-log plot of the execution time versus the ratio of the maximum to minimum propagation speed in the eikonal equation. We compare the Fast Marching Method and the Marching with a Correctness Criterion algorithm using a second-order, adjacent-diagonal scheme."
\image latex ExecutionTimeRatio2D2nd.pdf "Log-log plot of the execution time versus the ratio of the maximum to minimum propagation speed in the eikonal equation. We compare the Fast Marching Method and the Marching with a Correctness Criterion algorithm using a second-order, adjacent-diagonal scheme." width=0.65\textwidth
*/






//=============================================================================
//=============================================================================
/*!
\page hj_3d Extension to 3-D



In this section we extend the previous results in this chapter and
consider the eikonal equation \f$| \nabla u | f = 1\f$ in 3-D.  As
before, finite difference schemes that use three adjacent grid points
are not suitable for the MCC algorithm.  We will devise a scheme that
uses adjacent and diagonal neighbors to do the differencing.  We will
examine the performance of the adjacent-diagonal difference scheme and
then compare the execution time of the FMM to that of the MCC
algorithm.



<!---------------------------------------------------------------------------->
\section hj_3d_adjacent Adjacent-Diagonal Difference Schemes

We first consider the first-order, adjacent difference scheme.
The adjacent difference scheme in 3-D differences in the
three coordinate directions.  When a grid point becomes known, it uses
three formulas for updating the values of its adjacent neighbors.
First, only the solution at that known grid point is used.
Thus \c differenceAdj() computes the solution using a
single known adjacent neighbor.


\verbatim
differenceAdj(a):
  return a + dx / f \endverbatim


Second, in \c differenceAdjAdj() the solution at the labeled
grid point is computed using pairs of known adjacent solutions.
One of these is the grid point that was just determined to be known, the
other is a known grid point in an orthogonal direction.  In this function
we test whether the characteristic comes from the region between the
two grid points before we compute the solution.


\verbatim
differenceAdjAdj(a, b):
  if |a - b| <= dx / f:
    return (a + b + sqrt(2 dx^2 / f^2 - (a - b)^2 )) / 2
  return Infinity \endverbatim


Finally, in \c differenceAdjAdjAdj() the solution at the
labeled grid point is computed using triples of known adjacent
solutions.  One of these is the grid point that was just determined to
be known, the other two are known grid points in orthogonal
directions.  Here it is easiest to test that the characteristic comes
from the correct octant after we compute the solution.


\verbatim
differenceAdjAdjAdj(a, b, c):
  s = a + b + c
  discriminant = s^2 - 3 (a^2 + b^2 + c^2 - dx^2 / f^2 )
  if disc >= 0:
    soln = (s + sqrt(discriminant)) / 3
    if soln >= a and soln >= b and soln >= c:
      return soln
  return Infinity \endverbatim


We develop a correctness criterion for the above first-order, adjacent scheme.
We follow the same approach as for the correctness criteria in 2-D.
We will determine a lower bound on the solution at a labeled
grid point using the assumption that it has at least
one unknown neighbor.  If the predicted solution there is no larger
than this lower bound, then it must be correct.

Let \f$\mu\f$ be the minimum solution among the labeled grid points.
The correct solution at any unknown adjacent neighbors may be as low as
\f$\mu\f$.  We determine the smallest predicted solution using an unknown
neighbor.  The solution at the known neighbor may be as small as
\f$\mu - \Delta x\f$.  (If it were smaller, the solution at the labeled
grid point would be less than \f$\mu\f$.)
We obtain a lower bound by using one unknown neighbor with a value of \f$\mu\f$
and two known neighbors with values of \f$\mu - \Delta x\f$ in
the difference scheme.  This yields a predicted solution of
\f$\mu\f$ for the labeled grid point.  Thus \f$\mu\f$ is our
lower bound; any labeled grid point with solution less than or equal
to \f$\mu\f$ is correct.  This is the same correctness criterion used in
the Fast Marching Method.  As in 2-D, we see that the 3-D adjacent
difference scheme is not suitable for the MCC algorithm.

We introduce a new stencil in analogy with the 2-D adjacent-diagonal
difference scheme.  Again, instead of differencing only in the coordinate
directions, we consider differencing in the diagonal directions as well.
In 3-D, we adopt the terminology that \e diagonal directions lie
between two coordinate directions
and \e corner directions lie between three coordinate directions.
Thus an interior grid point has
6 adjacent, 12 diagonal and 8 corner neighbors.
The figure below first shows the adjacent stencil
that uses the 6 adjacent grid points.  On the left we show the points alone.
On the right we connect triples of points that are used in the adjacent finite
difference scheme.  If the characteristic passes through a triangle face,
then those three points are used to predict a solution.
The characteristic may approach through any of the 8 faces.
There are a total
of 26 ways in which the solution may be predicted: 6 ways from
a single adjacent point (the corner of a triangle face),
12 ways from a pair of adjacent points (the side of a triangle face) and
8 ways from a triple of adjacent points that form a triangle face.

The figure next shows a stencil that uses the
6 adjacent and 12 diagonal grid points
to increase the number of directions in which information can flow.
On the left we show the points alone.
On the right we connect triples of points that are used in the finite
difference scheme.  Again, if the characteristic passes through a triangle
face, then those three points are used to predict a solution.
The characteristic may approach through any of the 32 faces.
By differencing in an adjacent direction, we can approximate
\f$\partial u / \partial x\f$, \f$\partial u / \partial y\f$ or
\f$\partial u / \partial z\f$.  By differencing in a diagonal direction
we can determine the sum or difference of two of these.  There are a
total of 98 ways in which the solution may be predicted:
6 ways from a single adjacent point,
12 ways from a single diagonal point,
24 ways from an adjacent-diagonal pair,
24 ways from a pair of diagonal points,
24 ways from an adjacent-diagonal-diagonal triple
and 8 ways from a triple of diagonal points.


\image html Stencils6Pt18Pt.jpg "The 7-point, adjacent stencil and the 19-point, adjacent-diagonal stencil."
\image latex Stencils6Pt18Pt.pdf "The 7-point, adjacent stencil and the 19-point, adjacent-diagonal stencil." width=0.5\textwidth


Below we give the six functions for computing the predicted solution using
known neighbors of a labeled grid point.
\c differenceAdj() uses a single adjacent neighbor.
\c differenceDiag() uses a single diagonal neighbor.
\c differenceAdjDiag() uses an adjacent and a diagonal neighbor and
\c differenceDiagDiag() uses two diagonal neighbors.
These two functions check that the characteristic passes between the grid
points before computing the predicted solution.
\c differenceAdjDiagDiag() uses one adjacent and two diagonal
neighbors.  When testing that the characteristic passes through the
triangle face, it tests the two adjacent-diagonal sides before computing
the predicted solution and tests the diagonal-diagonal side after.
\c differenceDiagDiagDiag() uses three diagonal
neighbors.  It tests that the characteristic passes through the
triangle face after computing the predicted solution.


\verbatim
differenceAdj(a):
  return a + dx / f \endverbatim


\verbatim
differenceDiag(a):
  return a + sqrt(2) dx / f \endverbatim


\verbatim
differenceAdjDiag(a, b):
  if 0 <= a - b <= dx / (sqrt(2) f):
    return a + sqrt(dx^2 / f^2 - (a - b)^2 )
  return Infinity \endverbatim


\verbatim
differenceDiagDiag(a, b):
  if |a - b| <=  dx / (sqrt(2) f):
    return a + b + sqrt(6 dx^2 /f^2 - 3 (a - b)^2)
  return Infinity \endverbatim


\verbatim
differenceAdjDiagDiag(a, b, c):
  if a >= b and a >= c:
  discriminant = dx^2 / f^2 - (a - b)^2 - (a - c)^2
  if discriminant >= 0:
    soln = a + sqrt(discriminant)
    if soln >= 3 a - b - c:
      return soln
  return Infinity \endverbatim


\verbatim
differenceDiagDiagDiag(a, b, c):
  discriminant = 3 dx^2 / f^2 - (a - b)^2 - (b - c)^2 - (c - a)^2
  if discriminant >= 0:
    soln = (a + b + c + 2 sqrt(discriminant)) / 3
    if soln >= 3 a - b - c and soln >= 3 b - c - a and soln >= 3 c - a - b:
      return soln
  return Infinity \endverbatim



<!---------------------------------------------------------------------------->
\section hj_3d_performance Performance Comparison of the Finite Difference Schemes


<!---------------------------------------------------------------------------->
\subsection hj_3d_performance_test Test Problems

We consider three test problems which are analogous to the 2-D test
problems.  In each problem the distance is computed from a point or
set of points.  We consider cases in which the solution is smooth, the
solution has high curvature and the solution has shocks, i.e., the
solution is not everywhere differentiable.

Each grid spans the domain \f$[-1/2 .. 1/2]^3\f$.
For the smooth solution, the distance
is computed from a single point at \f$(-3/4, -3/4, -3/4)\f$.  Next the
distance is computed from a single point in the center of the grid.
The difference schemes will make the largest errors near the center
where the solution has high curvature.  For the third test problem,
the distance is computed from \f$26\f$ points on the
sphere of unit radius, centered at the origin.  There are shocks along
planes that are equidistant from two points.  This test case produces
shock planes at a variety of angles.





<!---------------------------------------------------------------------------->
\subsection hj_3d_performance_convergence Convergence

<b>Smooth Solution.</b>
We examine the convergence of the solution using the two difference
schemes.  The first graph below
shows the \f$L_1\f$
error versus the grid spacing for grids ranging in size from \f$10^3\f$ to
\f$400^3\f$.  We see that going from an adjacent stencil to an
adjacent-diagonal stencil reduces the error by about a factor of 2.
The \f$L_\infty\f$ error shows the same behavior.


\image html ErrorOutside3DL1.jpg "L_1 error for a smooth solution. Log-log plot of the error versus the grid spacing."
\image latex ErrorOutside3DL1.pdf "L_1 error for a smooth solution. Log-log plot of the error versus the grid spacing." width=\textwidth

\image html ErrorOutside3DLinf.jpg "Maximum error for a smooth solution. Log-log plot of the error versus the grid spacing."
\image latex ErrorOutside3DLinf.pdf "Maximum error for a smooth solution. Log-log plot of the error versus the grid spacing." width=\textwidth



The table below shows the numerical rate of convergence
for the difference schemes.
We see that for the smooth solution, both of these first-order schemes
have first-order convergence.

<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.982 <td> 0.969
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.989 <td> 0.979
</table>





<b>Solution with High Curvature.</b>
Next we examine the convergence of the solution with high curvature
using the two difference schemes.
The first graph below
shows the \f$L_1\f$ error versus the grid
spacing for grids ranging in size from \f$10^3\f$ to \f$400^3\f$.
Using an adjacent-diagonal stencil still
reduces the error by about a factor of 2.  The \f$L_\infty\f$ error
shows the same behavior.


\image html ErrorCenter3DL1.jpg "L_1 error for a solution with high curvature. Log-log plot of the error versus the grid spacing."
\image latex ErrorCenter3DL1.pdf "L_1 error for a solution with high curvature. Log-log plot of the error versus the grid spacing." width=\textwidth

\image html ErrorCenter3DLinf.jpg "Maximum error for a solution with high curvature. Log-log plot of the error versus the grid spacing."
\image latex ErrorCenter3DLinf.pdf "Maximum error for a solution with high curvature. Log-log plot of the error versus the grid spacing." width=\textwidth


The table below shows the numerical rate of convergence
for the two difference schemes.  For the solution with high curvature,
these first-order schemes have less than first-order convergence.  For
both the adjacent and the adjacent-diagonal schemes it is about \f$0.8\f$.

<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.799 <td> 0.809
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.807 <td> 0.816
</table>






<b>Solution with Shocks.</b>
Finally we examine the convergence of the solution with shocks using the two
difference schemes.  The first graph below
shows the \f$L_1\f$ error versus the grid spacing for grids ranging in
size from \f$10^3\f$ to \f$400^3\f$.  Yet again, using an adjacent-diagonal
stencil reduces the error by about a factor of 2.  The \f$L_\infty\f$
error shows the same behavior.




\image html ErrorSphereOutside3DL1.jpg "L_1 error for a solution with shocks. Log-log plot of the error versus the grid spacing."
\image latex ErrorSphereOutside3DL1.pdf "L_1 error for a solution with shocks. Log-log plot of the error versus the grid spacing." width=\textwidth

\image html ErrorSphereOutside3DLinf.jpg "Maximum error for a solution with shocks. Log-log plot of the error versus the grid spacing."
\image latex ErrorSphereOutside3DLinf.pdf "Maximum error for a solution with shocks. Log-log plot of the error versus the grid spacing." width=\textwidth


The table below shows the numerical rate of
convergence for the solution with shocks using the two difference schemes.
The convergence for both schemes is a little less than first-order.
The adjacent-diagonal scheme has a slightly higher rate of convergence.


<table>
<tr>
<th> Scheme <th> L_1 Error <th> Maximum Error
<tr>
<td> First-Order, Adjacent <td> 0.943 <td> 0.924
<tr>
<td> First-Order, Adjacent-Diagonal <td> 0.961 <td> 0.948
</table>





<!---------------------------------------------------------------------------->
\subsection hj_3d_performance_execution Execution Time


The figure below shows the
execution times for the two difference schemes using the Fast Marching
Method.  The distance from a center point is computed on grids ranging
in size from \f$10^3\f$ to \f$400^3\f$.  Using the adjacent-diagonal stencil
is more computationally expensive than using the adjacent stencil.
The execution time per grid point increases as the grid grows.
However, the performance difference between each of the methods
remains roughly constant.  This reflects the fact that the cost per
grid point of labeling does not depend on the grid size, but the cost
per grid point of maintaining the binary heap increases with the
increasing grid size.


\image html ExecutionTime3DFM.jpg "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching method with different stencils.  We show the algorithm with a first-order, adjacent scheme and a first-order, adjacent-diagonal scheme."
\image latex ExecutionTime3DFM.pdf "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching method with different stencils.  We show the algorithm with a first-order, adjacent scheme and a first-order, adjacent-diagonal scheme." width=\textwidth




<!---------------------------------------------------------------------------->
\section hj_3d_versus The FMM versus the MCC Algorithm



We compare the execution times of the MCC algorithm and the FMM
using the first-order, adjacent-diagonal scheme.  Again we implement a
method that measures the execution time of an ideal algorithm for
solving static Hamilton-Jacobi equations with an ordered, upwind
scheme.  The figure below shows the
execution times.  The performance of the MCC algorithm comes close to
that of the ideal algorithm.  None of these show exactly linear
scalability.  The execution time per grid point increases by \f$13 \%\f$
and \f$32 \%\f$ and the grid size varies from \f$10^3\f$ to \f$400^3\f$ for the
ideal algorithm and the MCC algorithm, respectively.  We believe this
effect is due to the increased cost of accessing memory as the memory
usage increases.  That is, with larger grids there will be more cache
misses so the cost of accessing a single grid value increases.  This
phenomenon has less of an effect on the ideal algorithm because it
does not access the solution or status arrays in determining which
grid points should become known.  We can see the \f$N \log N\f$ complexity
of the FMM, but it still performs well.  For the largest grid size,
its execution time is \f$88 \%\f$ higher than that of the MCC algorithm.


\image html ExecutionTime3D18Pt1st.jpg "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching algorithm, the marching with a correctness criterion algorithm and the ideal algorithm using the first-order, adjacent-diagonal scheme."
\image latex ExecutionTime3D18Pt1st.pdf "Log-log plot of the execution time per grid point versus the number of grid points for the fast marching algorithm, the marching with a correctness criterion algorithm and the ideal algorithm using the first-order, adjacent-diagonal scheme." width=0.8\textwidth
*/













//=============================================================================
//=============================================================================
/*!
\page hj_concurrent Concurrent Algorithm


Consider solving a static Hamilton-Jacobi equation on a shared-memory
architecture with \f$P\f$ processors.  It is easy to adapt the sequential MCC
algorithm to a concurrent one.  At each step in the algorithm the
minimum labeled solution is determined, the correctness criterion is
applied to the labeled set and then the correct grid points label their
neighbors.  Each of these operations may be done concurrently.
Each processor is responsible for \f$1/P\f$ of the labeled grid points.
The minimum labeled solution is found by having each processor examine
its share of the labeled grid points and then taking the minimum of the
the \f$P\f$ values.  Then each processor applies the correctness criterion
and labels neighbors of known grid points exactly as in the sequential
algorithm.  The computational complexity of the concurrent algorithm
is \f$\mathcal{O}(N/P + P)\f$.  (The \f$\mathcal{O}(P)\f$ term comes from taking the
minimum of the \f$P\f$ values to determine the minimum labeled solution and
from equitably dividing the labeled grid points.)
The Fast Marching Method is not amenable to this kind of concurrency
since only a single labeled grid point is determined to be correct at a time.


Next consider solving a static Hamilton-Jacobi equation on a
distributed-memory architecture with \f$P\f$ processors with the
MCC algorithm.  We distribute the solution grid over the processors.
The figure below
show a 2-D grid distributed over 16 processors.  In addition to its
portion of the grid, each processor needs to store a ghost boundary that
is as thick as the radius of the finite difference stencil.  That is,
it needs to store one extra row/column in each direction for a first
order difference scheme and 2 extra rows/columns for a second-order
difference scheme.  In updating the boundary grid points, each processor
communicates with up to four neighboring processors.  In each step of the
concurrent algorithm:
-# The processors communicate to determine the global value of the
minimum labeled solution.
-# Each processor communicates the grid points along the
edge of its grid that became known during the previous step.
-# Each processor performs one step of the sequential MCC algorithm.


<!--CONTINUE
\begin{figure}[h!]
  \label{figure distributed memory H-J}
  \begin{center}
    \setlength{\unitlength}{0.035\textwidth}

    %%
    %% poor load balancing
    %%
    \begin{picture}(16,12)(0,0)
      \put(0,0){\grid(12,12)(3,3)}
      %%
      \put(1.5,10.5){\raisebox{-3pt}{\makebox[0pt]{0}}}
      \put(4.5,10.5){\raisebox{-3pt}{\makebox[0pt]{1}}}
      \put(7.5,10.5){\raisebox{-3pt}{\makebox[0pt]{2}}}
      \put(10.5,10.5){\raisebox{-3pt}{\makebox[0pt]{3}}}
      \put(1.5,7.5){\raisebox{-3pt}{\makebox[0pt]{4}}}
      \put(4.5,7.5){\raisebox{-3pt}{\makebox[0pt]{5}}}
      \put(7.5,7.5){\raisebox{-3pt}{\makebox[0pt]{6}}}
      \put(10.5,7.5){\raisebox{-3pt}{\makebox[0pt]{7}}}
      \put(1.5,4.5){\raisebox{-3pt}{\makebox[0pt]{8}}}
      \put(4.5,4.5){\raisebox{-3pt}{\makebox[0pt]{9}}}
      \put(7.5,4.5){\raisebox{-3pt}{\makebox[0pt]{10}}}
      \put(10.5,4.5){\raisebox{-3pt}{\makebox[0pt]{11}}}
      \put(1.5,1.5){\raisebox{-3pt}{\makebox[0pt]{12}}}
      \put(4.5,1.5){\raisebox{-3pt}{\makebox[0pt]{13}}}
      \put(7.5,1.5){\raisebox{-3pt}{\makebox[0pt]{14}}}
      \put(10.5,1.5){\raisebox{-3pt}{\makebox[0pt]{15}}}
    \end{picture}
    %%
    %% better load-balancing
    %%
    \begin{picture}(12,12)(0,0)
      \linethickness{1pt}
      \put(0,0){\grid(12,12)(1,1)}
      \linethickness{2pt}
      \put(0,0){\grid(12,12)(4,4)}
      %%
      \matrixput(0.5,3.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{0}}}
      \matrixput(1.5,3.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{1}}}
      \matrixput(2.5,3.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{2}}}
      \matrixput(3.5,3.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{3}}}
      \matrixput(0.5,2.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{4}}}
      \matrixput(1.5,2.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{5}}}
      \matrixput(2.5,2.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{6}}}
      \matrixput(3.5,2.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{7}}}
      \matrixput(0.5,1.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{8}}}
      \matrixput(1.5,1.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{9}}}
      \matrixput(2.5,1.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{10}}}
      \matrixput(3.5,1.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{11}}}
      \matrixput(0.5,0.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{12}}}
      \matrixput(1.5,0.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{13}}}
      \matrixput(2.5,0.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{14}}}
      \matrixput(3.5,0.5)(4,0){3}(0,4){3}{\raisebox{-3pt}{\makebox[0pt]{15}}}
    \end{picture}

  \end{center}
  \caption{Left: A distribution of a 2-D grid over 16 processors.
    Each processor has one array.
    Right: A distribution that would better balance the load.  Each processor
    has 9 arrays.}
\end{figure}
-->


Consider a square 2-D grid with a first-order difference scheme.
The MCC algorithm will take \f$\mathcal{O}(R \sqrt{N})\f$ steps. Where \f$R\f$ is
the ratio of the highest to lowest propagation speed.
At each step, computing the global minimum labeled solution will take
\f$\mathcal{O}(\log P)\f$ communications.
During the course of the algorithm, a total of \f$4 (\sqrt{P} - 1) \sqrt{N}\f$
grid points are communicated to neighboring ghost boundary regions.
In addition, at each time step processors communicate the number of
grid points they will send to their neighbors.  Thus in each step,
each processor sends and receives 4 integers to determine how many
ghost grid points it will receive.  Then <em>on average</em> each processor
sends/receives \f$\mathcal{O}(\sqrt{P} / (R \sqrt{N})) = \mathcal{O}(1)\f$
grid points and computes correct
values for \f$\mathcal{O}(\sqrt{N}/(R P))\f$ grid points using the sequential
MCC algorithm.  This is a rosy picture except for the ``on average''
qualification.  With the data distribution shown on the left
we would expect a poor
load balance.  That is, we would expect that at any given point in
the computation some processors would have many labeled grid points while
others would have few or none.  Thus some processors would do many finite
difference computations while others would do none.  If the computation
were balanced, then the computational complexity would be
\f$\mathcal{O}(R \sqrt{N} \log P + N / P)\f$.


We could help balance the computational load by choosing a different
data distribution.  For the one shown on the right
we first divide the grid
into 9 sub-arrays.  Then we distribute the sub-arrays
as before.  Each processor still communicates with the same 4 neighbors,
but the total number of grid points communicated is increased to
\f$4 (\sqrt{P} \sqrt{M} - 1) \sqrt{N}\f$, where \f$M\f$ is the number of sub-arrays.
Each processor performs the sequential MCC algorithm on \f$M\f$ grids.
Let \f$L\f$ be the average load imbalance, where the load imbalance is defined
as the ratio of the maximum processor load to the average processor load.
If \f$P M \ll N\f$ then the computational complexity of the concurrent MCC
algorithm is \f$\mathcal{O}(R \sqrt{N} \log P + L N / P)\f$.


Note that any algorithm based on an adjacent difference scheme is
ill suited to a distributed-memory concurrent algorithm.  This is due
to the large numerical domain of dependence of these schemes.  A grid
point \f$(i,j,\ldots)\f$ with solution \f$s\f$ may depend on a grid point whose
solution is arbitrarily close to \f$s\f$ and whose location is arbitrarily
far away from \f$(i,j,\ldots)\f$.  (See \ref hj_mcc
for an example.)  This phenomenon would prevent concurrent algorithms
from having an acceptable ratio of computation to communication.
*/




//=============================================================================
//=============================================================================
/*!
\page hj_future Future Work


As introduced in the \ref shortest_paths_future section for the single-source
shortest-path problem, one could use
a cell array data structure to reduce the computational complexity of
the MCC algorithm.  This is the approach presented by Tsitsiklis.
He predicted
that this method would not be as efficient as his Dijkstra-like algorithm.
However, based on our experience with the MCC algorithm, we believe
that this approach would outperform the FMM as well.

We consider the eikonal equation \f$| \nabla u | f = 1\f$
in \f$K\f$-dimensional space, solved on the unit domain, \f$[0 .. 1]^K\f$.
Let the values of \f$f\f$ be in the interval \f$[A .. B]\f$ and let \f$R = B/A\f$.
Each cell in the cell array holds the labeled grid points with
predicted solutions in the interval
\f$[ n \frac{\Delta x}{\sqrt{2} B} .. (n+1) \frac{\Delta x}{\sqrt{2} B})\f$
for some integer \f$n\f$.  Consider the MCC algorithm in progress.  Let
\f$\mu\f$ be the minimum labeled solution.  The labeled distances are in
the range \f$[\mu .. \mu + \sqrt{2} \Delta x / A)\f$.  We define
\f$m = \lfloor \mu \frac{\sqrt{2} B}{\Delta x} \rfloor\f$. The first cell in the
cell array holds labeled grid points in the interval
\f$[ m \frac{\Delta x}{\sqrt{2} B} .. (m+1) \frac{\Delta x}{\sqrt{2} B})\f$.
By the correctness criterion, all the labeled grid points in this cell
are correct.  We intend to apply the correctness criterion only to the
labeled grid points in the first cell.  If they labeled their
neighbors, the neighbors would have predicted solutions in the interval
\f$[\mu + \frac{\Delta x}{\sqrt{2} B} ..
\mu + \frac{\Delta x}{\sqrt{2} B} + \frac{\sqrt{2} \Delta x}{A} )\f$.
Thus we need a cell array with \f$\lceil 2 R + 1 \rceil\f$ cells in
order to span the interval
\f$[\mu .. \mu + \frac{\Delta x}{\sqrt{2} B} + \frac{\sqrt{2} \Delta x}{A} )\f$,
which contains all the currently labeled solutions and the labeled
solutions resulting from labeling neighbors of grid points in the
first cell.  At each step of the algorithm, the grid points in the
first cell become known and label their neighbors.  If an unlabeled
grid point becomes labeled, it is added to the appropriate cell.  If a
labeled grid point decreases its predicted solution, it is moved to a
lower cell.  After the labeling, the first cell is removed and an
empty cell is added at the end.  Just as the FMM requires storing an array
of pointers into the heap of labeled grid points, this modification of
the MCC algorithm would require storing an array of pointers into the
cell array.

Now consider the computational complexity of the MCC algorithm that
uses a cell array to store the labeled grid points.  The complexity of
adding or removing a grid point from the labeled set is unchanged,
because the complexity of adding to or removing from the cell array is
\f$\mathcal{O}(1)\f$.  The cost of decreasing the predicted solution of a
labeled grid point is unchanged because moving a grid point in the
cell array has cost \f$\mathcal{O}(1)\f$.  We reduce the cost of applying
the correctness criterion from \f$\mathcal{O}(R N)\f$ to \f$\mathcal{O}(N)\f$
because each grid point is ``tested'' only once.  We must add the cost of
examining cells in the cell array.  Let \f$D\f$ be the difference between
the maximum and minimum values of the solution.  Then in the course of
the computation, \f$D/\frac{\Delta x}{\sqrt{2} B}\f$ cells will be examined.
The total computational complexity of the MCC algorithm with a cell
array for the labeled grid points is \f$\mathcal{O}(N + B D / \Delta x)\f$.
Note that for pathological cases in which a characteristic weaves
through most of the grid points, \f$D\f$ could be on the order of
\f$\Delta x N / A\f$.  (See the figure below for an
illustration of such a pathological case.)
In this case \f$B D / \Delta x \approx R N\f$ and the computational
complexity is the same as that for the plain MCC algorithm.  However,
for reasonable problems we expect that \f$D = \mathcal{O}(1 / A)\f$.  In
this case the computational complexity is \f$\mathcal{O}(N + R / \Delta x) =
\mathcal{O}(N + R N^{1 / K}) = \mathcal{O}(N)\f$.


\image html PathologicalWeave.jpg "A pathological case in which a characteristic weaves through most of the grid points.  We show a 9 by 9 grid.  Green grid points have a high value of f while red grid points have a low value.  If the initial condition were specified at the lower, left corner then there would be a characteristic roughly following the path of green grid points from the lower left corner to the upper right corner."
\image latex PathologicalWeave.pdf "A pathological case in which a characteristic weaves through most of the grid points.  We show a 9 by 9 grid.  Green grid points have a high value of f while red grid points have a low value.  If the initial condition were specified at the lower, left corner then there would be a characteristic roughly following the path of green grid points from the lower left corner to the upper right corner." width=0.3\textwidth
*/




//=============================================================================
//=============================================================================
/*!
\page hj_conclusions Conclusions



Sethian's Fast Marching Method is an efficient method for solving static
Hamilton-Jacobi equations.  It orders the finite difference operations
using a binary heap.  The computational complexity is \f$\mathcal{O}(N \log N)\f$
where \f$N\f$ is the number of grid points.  For grid sizes
of \f$10^6\f$, the cost of the finite difference operations (\f$\mathcal{O}(N)\f$)
is of the same order as the cost of the heap operations
(\f$\mathcal{O}(N \log N)\f$).

Applying the Marching with a Correctness Criterion methodology to solving
static Hamilton-Jacobi equations requires the use of adjacent-diagonal
finite difference schemes.  Typical upwind schemes difference in
adjacent (coordinate) directions.  Adjacent-diagonal schemes difference
in both adjacent and diagonal directions.  They reduce the numerical
domain of dependence and enable efficient correctness criteria to be
applied.  In addition, they offer greater accuracy, but are computationally
more expensive.

The Marching with a Correctness Criterion algorithm produces the same
solution as the Fast Marching Method, but with computational
complexity \f$\mathcal{O}(R N)\f$ where \f$R\f$ is the ratio of the highest to
lowest propagation speed.  Its execution times come close to those
of an ideal ordered, upwind, finite difference method.  (For an ideal
method, the cost of ordering the grid points would be negligible.)  In
practice, the MCC algorithm has modestly lower execution times than
the FMM and typically uses \f$1/4\f$ less memory.  For a grid size
of \f$10^6\f$, the MCC algorithm executes in about half the time of the FMM.
The computational complexity of the MCC Algorithm can be reduced to
\f$\mathcal{O}(N)\f$  by using a cell array to store the labeled grid points.
*/

} // namespace hj
}

#define __hj_hj_ipp__
#include "stlib/hj/hj.ipp"
#undef __hj_hj_ipp__

#endif

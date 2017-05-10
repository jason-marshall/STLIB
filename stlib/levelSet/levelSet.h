// -*- C++ -*-

#if !defined(__levelSet_h__)
#define __levelSet_h__

#include "stlib/levelSet/solventExcluded.h"

namespace stlib
{
//! Classes and functions for level set calculations.
namespace levelSet
{
}
}

/*!
\mainpage Level Sets

<!------------------------------------------------------------------------->
\section levelSetIntroduction Introduction

\par
Consult \ref levelSetSethian1999 "Level Set Methods and Fast Marching Methods"
and \ref levelSetOsher2003 "Level Set Methods and Dynamic Implicit Surfaces."
for an overview of level set methods.

<!------------------------------------------------------------------------->
\section levelSetCode Code

\par Classes.
The levelSet::GridUniform class is a grid with uniform spacing.
The levelSet::Grid class uses a single level of adaptive mesh refinement
(AMR). The patches in this grid are elements of the levelSet::Patch class. The
levelSet::GridVirtual class has information about patches, but does not
store them.

\par Dense grids.
Using uniform grids has the advantage that the algorithms are easier to
implement. The disadvantage is the large amount of storage required. For
most applications, the level set function is only needed in a narrow band
around the zero iso-surface. With a dense grid, one stores values for the
level set function throughout the rectilinear domain. Because of storage
issues, I use the dense grid implementation primarily for verifying the
correctness of the AMR implementations.

\par AMR grids.
The AMR grid implemented in levelSet::Grid uses non-overlapping patches,
implemented in levelSet::Patch. For most algorithms, the extents of the patch
may be set arbitrarily. However, some algorithms, like marking the grid
points which are outside an object with markOutsideAsNegativeInf(), require
that the patch extents match the number of bits in in a character, namely 8.
This is because the method uses bit twiddling for vector operations.
Follow the link for details. In any case, an extent of 8 is typically
a good choice for 3-D applications. 
Patches may not be large because one needs to capture the narrow band without
wasting a lot of storage. But if patches are too small, then computing
the level set function on the patch is an insignificant amount of work.
The overhead of scheduling patches may dominate.
An extent of 8 results in 512 grid points per patch, which is
usually a good compromise.

\par Virtual AMR grids.
A <em>virtual</em> AMR grid, implemented in levelSet::GridVirtual, differs
in two respects from the regular AMR grid:
-# The virtual grid stores information about the patches, but does not store
the patches themselves. This is useful for algorithms in which the level
set may be computed and processed one patch at a time.
-# The patches overlap at the upper boundaries. This is necessary to process
the level set function. Note that when using Grid, the level set is stored
on the non-overlapping patches. But when one uses the
\ref levelSetMarchingSimplices "marching simplices algorithm", one works
with the overlapping voxel patches, instead of the non-overlapping
vertex patches.

\par
Virtual AMR grids have the the advantage that the storage requirements
are minimal. One only works with a single patch at a time. The downside
is that because the patches overlap, a significant portion of the
grid points are duplicated. For a patch extent of 8 in 3-D, each
patch has 512 grid points, 169 of which are duplicates.
(The number of duplicates is the number of grid points along the
upper boundary. 8<sup>3</sup> - 7<sup>3</sup> = 169.) Thus, working
with a virtual grid is typically more expensive.
Another limitation of virtual grids is that some level set operations
require the global level set function (flood filling operations are
one example of this).

\par GPGPU.
Some methods have been ported to use the GPU via CUDA. These methods use
AMR mesh. Indeed, efficient GPU utilization was a factor in designing
the AMR data structure. The patches naturally correspond to CUDA blocks
and the grid points may be assigned to threads. Furthermore, the storage
for all of the patches is contiguous in memory. This makes it easy to
transfer an AMR grid to and from the GPU.

\par Functions.
There are functions for performing the following calculations.
- \ref levelSetBoolean "Boolean" operations.
- \ref levelSetCount "Count" the number of known or unknown values in a grid.
- \ref levelSetFlood "Flood fill" operations to set unknown values to a
signed constant.
- Level sets related to the \ref levelSetSolventExcluded "solvent excluded surface".
- Level sets for the \ref levelSetPositiveDistance "positive"
and \ref levelSetNegativeDistance "negative" distance to the union of a
set of balls.

\par Special values.
Use NaN (Not a Number, with value \c std::numeric_limits<T>::quiet_NaN())
to denote an unknown value. Use <tt>x != x</tt> to test whether the
variable \c x is NaN. This curious predicate arises from the axiom that NaN is
not equal to any other number, including itself. Use \f$\infty\f$,
with value \c std::numeric_limits<T>::infinity(), to
denote large unknown positive distances. Use \f$-\infty\f$ for large
unknown negative distances.

<!------------------------------------------------------------------------->
\section levelSetMolecules Modelling Molecules

\par Solvent-Accessible Manifold.
Each atom in a molecule may be modeled as a ball. For each, the radius is
determined by the
<a href="http://en.wikipedia.org/wiki/Van_der_Waals_radius">van der Waals
volume</a>. The <em>van der Waals manifold</em> is the union of the balls.
The solvent water molecule is modeled with a single ball. The radius is
typically 1.4 Angstroms. Consider a solvent ball rolling over the surface
of the van der Waals manifold. The
<a href="http://en.wikipedia.org/wiki/Accessible_surface_area">
solvent-accessible manifold</a> is bounded by the locus points of the center
of the solvent. Equivalently, it is the union of the molecular balls where
each has been expanded by the solvent radius.

\par Solvent-Excluded Manifold.
The solvent ball is in a valid position if it does not intersect any of the
molecular balls. The solvent-excluded manifold is the negation of the union
of all valid solvent positions. Equivalently, one may construct it by
offsetting the surface of the solvent-accessible manifold by the negative
of the solvent radius. (Of course, when doing this one must account for
self-intersections and take the entropy-satisfying solution.)

\par Cavities.
The <em>solvent-excluded cavities</em> are the difference between the
solvent-excluded manifold and the van der Waals manifold. These are gaps
between the atoms where the solvent cannot reach. Consider a solvent that
is restricted to valid (non-intersecting) positions. It is <em>trapped</em> if
there is no continuous path to a position outside the convex hull of the
van der Waals manifold. The union of all trapped solvents forms the
<em>solvent-accessible cavities</em>. Roughly speaking, these are the cavities
where solvent molecules may be trapped.

<!------------------------------------------------------------------------->
\section levelSetRelated Related Work

\par
\ref levelSetCan2006 "Efficient molecular surface generation using level-set methods."
The authors use level-set methods to calculate molecular surfaces and cavities.
To calculate the SES, they start by seeding the distance at the atom centers.
The distance is propogated outward using the Fast Marching Method. Instead of
using finite differencing, they propagate the index of the closest atom center
and perform Euclidean distance calculations. This yields an approximation of
the true distance. The distance is calculated back inwards the distance of
the probe radius to obtain the SES. The level-set methods are carried out
on a dense, uniform grid. This, along with a rough characterization of
the surface, limits the accuracy of the method.

\par
\ref levelSetDias2010 "CUDA-based Triangulations of Convolution Molecular Surfaces."
The authors use the GPU to compule Blinn surfaces for molecules. The level set
for the surface is computed on the GPU with one thead per atom. Then they
use a Marching Cubes algorithm with linear interpolation on the GPU to
triangulate the surface.


<!------------------------------------------------------------------------->
\section levelSetImplementation Implementation Notes

\par
We use the class \c container::StaticArrayOfArrays to represent ball-patch
dependencies. That is, we use it as a sparse array, with a row for each patch.
The elements of each row are ball indices for the balls that affect the
patch. We use the 32-bit <tt>unsigned int</tt> number type for the array. This
allows us to use the same functions for the CPU and GPU implementations.

*/

/*!
\page levelSetBibliography Bibliography

<!--------------------------------------------------------------------------->
-# \anchor levelSetSethian1999
J. A. Sethian. "Level Set Methods and Fast Marching Methods" Cambridge.
<!--------------------------------------------------------------------------->
-# \anchor levelSetOsher2003
Stanley Osher, Ronald Fedkiw. "Level Set Methods and Dynamic Implicit
Surfaces." Springer.
<!--------------------------------------------------------------------------->
-# \anchor levelSetCan2006
Tolga Can, Chao-I Chen, and Yuan-Fang Wang.
"Efficient molecular surface generation using level-set methods."
Journal of Molecular Graphics & Modelling, Vol. 25, Issue 4, Pages 442-454,
Dec. 2006.
<!--------------------------------------------------------------------------->
-# \anchor levelSetDias2010
Sergio Dias, Kuldeep Bora, and Abel Gomes.
"CUDA-based Triangulations of Convolution Molecular Surfaces."
HPDC'10, June 20–25, 2010, Chicago, Illinois, USA.
*/
#endif

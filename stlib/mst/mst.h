// -*- C++ -*-

#if !defined(__mst_h__)
#define __mst_h__

#include "stlib/mst/MolecularSurface.h"
#include "stlib/mst/readXyzr.h"

namespace stlib
{
//! All classes and functions in the MST package are defined in the mst namespace.
namespace mst
{

/* CONTINUE
  \section mstToDoList To Do List

  - Add the functionality to pre-triangulate the plug-in proteins.  The
  plug-in (and its triangle mesh) would be rotated and translated to the
  correct position.  There, it will be clipped by the macro-molecule and
  vice versa.  This should significantly improve the speed.
  - Keep track of the triangles that have not been modified from the original
  tesselation of the sphere.  When a patch is modified, only some of its
  triangles are modified.  One can compare the new patch with the old.
  This capability would reduce the modified triangle count when plugging in
  proteins at the cost of some overhead.
  - Analyze the quality of the triangulation.
    - What is the quality distribution of the triangles?
    - How well does the triangulation match the actual surface?
*/



/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\mainpage Molecular Surface Triangulation

This package implements a method for efficiently triangulating the surface
of a dynamically changing macro-molecule.  The triangulation is used in
protein design in which one uses boundary integral equations to solve
molecular electrostatic problems.  There are two requirements for the
triangulation: Firstly, the triangulation must have user-specified accuracy.
That is, one can roughly specify the number of triangles.  Secondly,
inserting or removing a small number of atoms must result in small changes
to the triangulation.


<!--------------------------------------------------------------------------->
\section mst_recent Recent Developments.

The mst::MolecularSurface::updateSurface() function now uses hybrid clipping,
a mix of rubber clipping and cut clipping.  There are similar functions
for that use either only rubber clipping or only cut clipping.
The \ref mst_driver "MST driver" uses the hybrid clipping method by default.

If you don't want very small
triangles in the mesh, use mst::MolecularSurface::setMinimumAllowedArea()
to specify a minimum allowed area for the triangles.
For the \ref mst_driver "MST driver", you can set the minimum allowed area
with the \c -area option.


<!--------------------------------------------------------------------------->
\section mst_currentCapabilities Current Capabilities.

The triangulation is performed with the mst::MolecularSurface class.
One can dynamically insert and erase atoms.  At any point, one can request
the surface triangulation to be updated.  Following this, one can access
the triangles that have been modified.  See the class documentation
for details.


<!--------------------------------------------------------------------------->
\section mst_drivers Example Programs.


The \ref mst_driver "MST driver" exposes most of the functionality in the
mst::MolecularSurface class.  One can generate triangulations and test the
dynamic capability.
I also have a \ref mst_allSurfaces program that meshes all the surfaces
(not the visible surface) of a molecule.  This is only useful testing or
visualization purposes.

There are a number of programs that are useful in visualizing and analyzing
the triangulations:
- \ref examples_geom_mesh_utility_iss2vtk "iss2vtk" converts triangle mesh
  file to a VTK file.  It accepts a triangle mesh and attributes defined
  on the triangle faces as input.
- \ref examples_geom_mesh_utility_cellAttributes "cellAttributes"
  is useful for visualizing the quality of triangle meshes.
- \ref mst_signedDistance "signedDistance" computes the signed distance to
  the molecular surface for a set of points.  This can be used to determine
  how well the triangulation approximates the molecular surface.
- \ref mst_subdivideSphericalTriangles "subdivideSphericalTriangles"
  subdivide a mesh of spherical triangles.  This is useful when you want to
  visualize a triangle mesh interpreted as spherical triangles.


<!--------------------------------------------------------------------------->
\section mst_usageExamples Usage Examples.

- \ref mst_gen


<!--------------------------------------------------------------------------->
\section mstMolecularSurfaces Molecular Surfaces.

There are several surfaces that one can use to describe a molecule.
Let each atom be modeled with a ball that has a specified position and
radius.   The <em>union-of-balls</em> surface is the visible portion of
the atomic surfaces.
This surface is composed of patches of spheres.
One can offset this surface by a probe radius to obtain
a <em>solvent excluded surface</em>.
This is equivalent to the molecular surface
where the radius of each atom is increased by the probe radius.
This package can triangulate the molecular surface (and hence
an offset surface).

Another
possibility is the <em>rolling ball surface</em>.  This is the envelope of
the surface of a spherical probe as it rolls over the surface.
This surface is composed of patches of spherical caps, tori,
and spherical cups.  When the rolling ball is in contact with a single atom,
it produces a spherical cap (The radius is that of the atom).
When it is in contact with two atoms, it produces a toroidal patch.
When it is in contact with three atoms, it produces a spherical cup.





<!--------------------------------------------------------------------------->
\section mstTriangulationRequirements Dynamic Triangulation Requirements.

We are interested in generating triangulations for dynamically changing
molecules.  The triangulation must permit efficient insertion and
removal of atoms.  (Moving an atom is equivalent to removing it and
inserting it in a new position.)
More precisely, if one changes a small number of atoms, a small number
of triangles should be modified.  To this end, each triangle is associated
with a particular atom.  The visible portion of each atom is triangulated.
This enables efficient updating of the triangulation when atoms have been
inserted and/or removed.  The affected atoms remove their portions of the
triangulation and then re-triangulate their visible surfaces.





<!--------------------------------------------------------------------------->
\section mstTriangulationAlgorithm The Triangulation Algorithm.

Consider the triangulation of a surface.
Ideally, one wants most of the triangles to be nearly equilateral.
One might want the triangles to have uniform sizes or sizes that match
the local feature size of the surface.  For a uniform-size mesh, one could
specify a target edge length.  For the other case, one could specify a
target edge length in terms of the local feature size.  For example, one
might constrain that the edge length be proportional to the radius of
curvature.

Our approach to meshing the visible surface of a molecule is based on
generating meshes for the entire surfaces of each atom and then clipping
the meshes to get rid of the hidden portions of the surface.  We describe
these two steps in the sections below.



<!--- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection mstTriangulatingAtoms Triangulating Atoms.

To generate a triangle
mesh for the visible portion of an atom, we start by tesselating the atom
with equilateral (or nearly equilateral) triangles.  One can tesselate a
sphere with a tetrahedron (4 triangles), an octahedron (8 triangles),
or an icosahedron (20 triangles).
We use the latter two in the triangulation algorithm; they are shown below.
We do not use the tetrahedron because it is too coarse.
Below we indicate the maximum edge lengths for the tesselation of a sphere
with unit radius.

\image html octahedron.jpg "The octahedron.  (8 triangles.  Maximum edge length is 1.41.)"
\image latex octahedron.pdf "The octahedron.  (8 triangles.  Maximum edge length is 1.41.)"

\image html icosahedron.jpg "The icosahedron.  (20 triangles.  Maximum edge length is 1.05.)"
\image latex icosahedron.pdf "The icosahedron.  (20 triangles.  Maximum edge length is 1.05.)"

We can use uniform subdivision to obtain other high quality meshes.
In uniform subdivision, each edge is split and each triangle is replaced
by four.  After subdividing, we move the new vertices to lie on the sphere.
Thus we can generate meshes with either \f$8 \times 4^n\f$ or
\f$20 \times 4^n\f$ triangles for \f$n = 0, 1, 2, \ldots\f$.  The first
subdivisions of the octahedron and the icosahedron are shown below.
(We do not use subdivisions of the tetrahedron because of the relatively
low quality of the triangles in the resulting meshes.)

\image html octahedronSubdivided.jpg "The subdivided octahedron.  (32 triangles.  Maximum edge length is 1.)"
\image latex octahedronSubdivided.pdf "The subdivided octahedron.  (32 triangles.  Maximum edge length is 1.)"

\image html icosahedronSubdivided.jpg "The subdivided icosahedron.  (80 triangles.  Maximum edge length is 0.618.)"
\image latex icosahedronSubdivided.pdf "The subdivided icosahedron.  (80 triangles.  Maximum edge length is 0.618.)"

Now we turn to the task of selecting an appropriate mesh to triangulate
the surface of a particular atom.   A simple approach is to select a level
of refinement to use for each atom in the mesh.  We define the octahedron
to be level 0 and the icosahedron to be level 1.

Below we show a few examples with different levels of refinement.  We have a
"molecule" composed of four disjoint atoms of radius 1, 2, 3 and 4 Angstroms.
We generate triangulations with levels 0, 1, 2, and 3.
The following commands, which generate the VTK file for the first figure,
are executed in \c stlib/examples/mst.  We use \c mst.exe to triangulate
the surface of the molecule.  We compute the centroids of the cells in
the mesh with \c computeCentroids32.exe.  Next we compute the signed distance
from the centroids to the visible surface.  Finally, we make a VTK file
containing the triangle mesh and the distance data.

\verbatim
./mst.exe -level=0 -radius=0 ../../data/mst/fourDisjoint1234.xyzr fourLevel0.txt
../geom/mesh/utility/computeCentroids32.exe fourLevel0.txt centroids.txt
./signedDistance.exe four.xyzr centroids.txt distance.txt
../geom/mesh/utility/iss2vtk32.exe -cellData=distance.txt fourLevel0.txt fourLevel0.vtu \endverbatim

\image html fourLevel0.jpg "Triangulation with refinement level 0.  (Octahedron.)"
\image latex fourLevel0.pdf "Triangulation with refinement level 0.  (Octahedron.)"

\image html fourLevel1.jpg "Triangulation with refinement level 1.  (Icosahedron.)"
\image latex fourLevel1.pdf "Triangulation with refinement level 1.  (Icosahedron.)"

\image html fourLevel2.jpg "Triangulation with refinement level 2.  (Subdivided octahedron.)"
\image latex fourLevel2.pdf "Triangulation with refinement level 2.  (Subdivided octahedron.)"

\image html fourLevel3.jpg "Triangulation with refinement level 3.  (Subdivided icosahedron.)"
\image latex fourLevel3.pdf "Triangulation with refinement level 3.  (Subdivided icosahedron.)"

By showing the signed distance from the triangle centroids to the molecular
surface, we indicate how well the triangulation matches the surface.  We
could define the error in the approximating the surface by integrating
the magnitude of the distance to the triangulation over the surface.  Because
the triangulation is a linear approximation of the surface, the error is
roughly proportional to the square of the triangle edge length.

With a fixed level of refinement for all atoms, the mesh will deviate
more from the surface for larger atoms.  However, the relative
deviation (with respect to the radius of curvature) is the same for
all atoms.  Thus, this simple approach is sensible.



Now instead of a mesh with a uniform level of refinement for each atom,
consider a mesh that has approximately uniform triangle sizes.
The triangle size directly impacts the quality of the
triangulation and the work that must be done to perform a calculation on
the mesh.  Smaller triangles mean higher quality, but more work.  Balancing
accuracy and computational cost considerations would give one a desired
triangle size, and hence a desired edge length.  This target edge length
determines the appropriate initial mesh for each atom.  We simply select
the coarsest mesh whose maximum edge length is no greater than the maximum
allowed edge length.

Below we show a few examples of the effect of edge length.  We use the
same "molecule" as above.  We generate triangulations with maximum
allowed edge lengths of 4, 2 and 1 Angstrom.  The following commands
generate the VTK file for the first figure.

\verbatim
./mst.exe -length=0,4 -radius=0 ../../data/mst/fourDisjoint1234.xyzr four4.txt
../geom/mesh/utility/computeCentroids32.exe four4.txt centroids.txt
./signedDistance.exe four.xyzr centroids.txt distance.txt
../geom/mesh/utility/iss2vtk32.exe -cellData=distance.txt four4.txt four4.vtu \endverbatim

\image html four4.jpg "Triangulation with a target edge length of 4 Angstroms."
\image latex four4.pdf "Triangulation with a target edge length of 4 Angstroms."

\image html four2.jpg "Triangulation with a target edge length of 2 Angstroms."
\image latex four2.pdf "Triangulation with a target edge length of 2 Angstroms."

\image html four1.jpg "Triangulation with a target edge length of 1 Angstrom."
\image latex four1.pdf "Triangulation with a target edge length of 1 Angstrom."

We see that by specifying a fixed maximum edge length, larger atoms are better
approximated than smaller atoms.  This is quite different than the results
we obtained by using a fixed level of refinement.  Which approach is better
depends on how one is going to use the mesh.  For some applications, one
might want the triangle size to be proportional to the local feature size
of the surface.  For other applications, the local feature size might not be
important and a mesh with uniform-sized triangles would be appropriate.
With these considerations in mind, we chose to specify the maximum allow
edge length as a linear function of the atomic radius.  In generating
the initial mesh for an atom of radius r, the maximum allowed edge length
is \f$a r + b\f$, where a and b are user specified constants.  By choosing
\f$a = 0\f$, one can specify a (roughly) uniform triangle size.
Choosing \f$a = 1\f$ is equivalent to a uniform level of refinement.

In the figures below, we show a transition from uniform triangle size to a
uniform level of refinement.  We choose values of \f$a\f$ and \f$b\f$ such
that \f$a + b = 1\f$.  Thus the atom with unit radius always has the same
triangulation, which happens to be a subdivided octahedron.  Because the
uniform size is small, as we move toward a uniform level of refinement,
the number of triangles decreases.
Note: Although the maximum edge length functions are different,
the fourth and fifth triangulations are the same.  This is because the
triangulations change in discrete steps.


\image html fourA0.0B1.0.jpg "Triangulation with a maximum allowed edge length of 0 * r + 1 Angstroms.  1952 triangles."
\image latex fourA0.0B1.0.pdf "Triangulation with a maximum allowed edge length of 0 * r + 1 Angstroms.  1952 triangles."

\image html fourA0.2B0.8.jpg "Triangulation with a maximum allowed edge length of 0.2 * r + 0.8 Angstroms.  800 triangles."
\image latex fourA0.2B0.8.pdf "Triangulation with a maximum allowed edge length of 0.2 * r + 0.8 Angstroms.  800 triangles."

\image html fourA0.4B0.6.jpg "Triangulation with a maximum allowed edge length of 0.4 * r + 0.6 Angstroms.  560 triangles."
\image latex fourA0.4B0.6.pdf "Triangulation with a maximum allowed edge length of 0.4 * r + 0.6 Angstroms.  560 triangles."

\image html fourA0.6B0.4.jpg "Triangulation with a maximum allowed edge length of 0.6 * r + 0.4 Angstroms.  272 triangles."
\image latex fourA0.6B0.4.pdf "Triangulation with a maximum allowed edge length of 0.6 * r + 0.4 Angstroms.  272 triangles."

\image html fourA0.8B0.2.jpg "Triangulation with a maximum allowed edge length of 0.8 * r + 0.2 Angstroms.  272 triangles."
\image latex fourA0.8B0.2.pdf "Triangulation with a maximum allowed edge length of 0.8 * r + 0.2 Angstroms.  272 triangles."

\image html fourA1.0B0.0.jpg "Triangulation with a maximum allowed edge length of 1 * r + 0 Angstroms.  128 triangles."
\image latex fourA1.0B0.0.pdf "Triangulation with a maximum allowed edge length of 1 * r + 0 Angstroms.  128 triangles."




<!--- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection mstClipping Clipping.

We have dealt with triangulating the whole surface of an atom.  Now we turn our
attention to triangulating the visible portion of the surface.  The visible
portion of a specified atom is what remains after every other atom in the
molecule has clipped the surface.  When an atom y clips an atom x, the portion
of the surface of x that is in the interior of y is removed.

Two approaches to clipping a triangle mesh are <em>rubber clipping</em>
and <em>cut clipping</em>.  First consider rubber clipping.  When an edge of
the mesh crosses the clipping curve, one of the two vertices of the edge
are moved onto that curve.  After this, one removes the triangles that
are outside the clipping curve.  The approach is called rubber clipping
because moving the vertices stretches the edges.  Next consider cut clipping.
As the name suggests, the clipping curve cuts off portions of triangles
which intersect it.  When a triangle is cut, either a triangle or
quadrilateral remains.  In the case of a quadrilateral, an edge is inserted
to form two triangles.

There are benefits and potential problems associated with each approach.
With rubber clipping, one is able to maintain a small number of triangles.
The downside is that one may lose accuracy along the clipping curves if the
shape of the visible portion is complicated.  Cut clipping gives a more
accurate representation of the surface, but introduces more triangles.  As one
would expect from arbitrarily cutting a triangle, some of the cut triangles
have very poor quality.  One can ameliorate these problems by using mesh
optimization techniques.  The mst::MolecularSurface class implements both
of these approaches.  Use mst::MolecularSurface::updateSurface() for
rubber clipping and mst::MolecularSurface::updateSurfaceWithCutClipping()
for cut clipping.

Below we triangulate a surface using an edge length function of 0.6 r.
We try both rubber clipping and cut clipping.

\verbatim
./mst.exe -rubber -length=0.6,0 ../../data/mst/gen.xyzr genA0.6B0Rubber.txt
./mst.exe -cut -length=0.6,0 ../../data/mst/gen.xyzr genA0.6B0Cut.txt \endverbatim

\image html genA0.6B0Rubber.jpg "Rubber clipping.  17,571 triangles.  Mean modified condition number is 0.76."
\image latex genA0.6B0Rubber.pdf "Rubber clipping.  17,571 triangles.  Mean modified condition number is 0.76." width=\textwidth

\image html genA0.6B0Cut.jpg "Cut clipping.  41,312 triangles.  Mean modified condition number is 0.59."
\image latex genA0.6B0Cut.pdf "Cut clipping.  41,312 triangles.  Mean modified condition number is 0.59." width=\textwidth

Although cut clipping yields a more accurate representation of the surface,
it introduces a lot of triangles.  The cut-clipped mesh has
almost twice as many triangles as the rubber-clipped mesh.  Also, we see that
many of the cut triangles have poor quality.


A hybrid approach that combines rubber clipping and cut clipping
merges the strengths of the two methods.  We use rubber clipping
when the vertex is close the clipping plane and cut clipping for the
other cases.  With hybrid clipping we try to keep the low triangle count
of rubber clipping and obtain the higher accuracy of cut clipping.

\verbatim
./mst.exe -length=0.6,0 ../../data/mst/gen.xyzr genA0.6B0Hybrid.txt \endverbatim

\image html genA0.6B0Hybrid.jpg "Hybrid clipping.  21,599 triangles.  Mean modified condition number is 0.71."
\image latex genA0.6B0Hybrid.pdf "Hybrid clipping.  21,599 triangles.  Mean modified condition number is 0.71." width=\textwidth

When inserting an atom, we efficiently identify the atoms that might
clip its mesh by using an orthogonal range query data structure.
Also, each atom keeps track of which atoms it has clipped.  When
erasing an atom, this allows us to quickly determine the neighboring
atoms that will need remeshing.  The triangles of the surface mesh are
stored in an array data structure that allows efficient insertion and
removal of elements.  It accomplishes this by storing the holes (empty
elements) in the array.  For the hybrid clipping example above, the
mesh was generated in 0.7 seconds on a 3 GHz Pentium 4.  Deleting and
inserting an atom takes about 12 milliseconds on average for that
example.


<!--------------------------------------------------------------------------->
\section mstMeshQuality Mesh Quality.


<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection mstTriangleSize Triangle Size.


Obviously, the triangulation should be a good approximation of the molecular
surface.  Because we will perform computations on the mesh, an additional
requirement is that it have a low triangle count.
These are conflicting goals.  As one lowers the triangle count, the accuracy
of the triangulation is reduced.

There are two sources of error in triangulating the molecular surface.
We name these the <em>surface error</em> and the <em>boundary error</em>.
Although, the vertices of each triangle lie on the molecular surface, the
face does not.  The triangle face is flat, whereas the corresponding portion
of the molecular surface is spherical.  (See the \ref mstSphericalTriangles
section for a way to address this issue.)  The the second source of error
occurs along the intersection curves between atoms.  The visible portion
of a single atom is a portion of a sphere.  The boundary of this surface
is where it intersects the surfaces of other atoms.  In the course of
triangulating the surface, this boundary is approximated.  Since the atoms
are meshed independently, the mesh is not conformal along the intersection
curves.

We saw above how the edge length affects the surface error by triangulating
spheres.  Clipping to triangulate the visible surface of an atom affects the
boundary error.  If the initial triangulation of a sphere is coarse, the
triangulation of its visible surface may have poor accuracy.  This is
particularly true when using rubber clipping.  The virtue of rubber clipping
is that it maintains a low triangle count.  When the triangulation is coarse,
it does not have enough flexibility to accurately represent a complicated
shape.

Below we show triangulations of the generic molecule with a range of target
edge lengths.  As above, we visualize the signed distance from the triangle
centriods to the molecular surface.


\image html distanceGen2.jpg "Triangulation with a target edge length of 2 Angstroms.  10,504 triangles."
\image latex distanceGen2.pdf "Triangulation with a target edge length of 2 Angstroms.  10,504 triangles."

For the target edge length of 2 Angstroms, the small atoms are represented very
poorly.


\image html distanceGen1.jpg "Triangulation with a target edge length of 1 Angstrom.  28,961 triangles."
\image latex distanceGen1.pdf "Triangulation with a target edge length of 1 Angstrom.  28,961 triangles."

For the target edge length of 1 Angstrom, the small atoms are reasonably
represented, but we can see the distorting effects of the clipping.


\image html distanceGen0.5.jpg "Triangulation with a target edge length of 0.5 Angstroms.  101,349 triangles."
\image latex distanceGen0.5.pdf "Triangulation with a target edge length of 0.5 Angstroms.  101,349 triangles."

When we reduce the target edge length to 0.5, the mesh is very accurate, but
it has an awful lot of triangles.




<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection mstTriangleShape Triangle Shape.

Note that the visible portion of an atom may have small angles or thin
strips.  In many cases, it is not possible to triangulate such a region
with a small number of well-shaped triangles.  Thus, some poorly-shaped
triangles are inevitable.

Roughly speaking, a triangle can have poor quality because of a small angle
or a large angle (which then implies that the other two angles are small).
Small angles are bad because they are wasteful.  The triangle covers little
area of the surface.  Large angle are also wasteful, but they also diminish
the accuracy of the triangulation.  Geometrically, the triangle normal may
differ greatly from the surface normal.  If a field is interpolated on
the triangulation, large angles lead to large errors in the derivative
of the field.

Small angles are unavoidable.  Thus, the solver which uses the triangulation
must be able to handle them robustly.  A large angle may be removed by
splitting the opposite edge.  This introduces two triangles with small angles
in place of the one triangle with a large angle.  (This is currently not
done.)





<!--------------------------------------------------------------------------->
\section mstSphericalTriangles Spherical Triangles


<!--CONTINUE: Cite spherical triangles.  A graphic would be useful.-->

Because each triangle lies on a sphere describing a single atom, it may be
interpreted as a spherical triangle.  That is, the face of the triangle
lies on the sphere and the edges of the triangle are arcs of great circles.
This interpretation of the mesh is clearly more accurate
than the flat-triangle mesh.  With spherical triangles, each triangle exactly
matches a portion of the molecule surface.  Thus this removes the surface
error.  Only the boundary error, which occurs along the
intersection curves between atoms, remains.

One can use either the mst::MolecularSurface::getAtomIdentifierForTriangle()
or the mst::MolecularSurface::getAtomForTriangle() member functions to
access the atom associated with a particular triangle in the mesh.
One can use the mst::Atom::getRadius() and mst::Atom::getCenter() member
functions to get the geometry of an atom.

There are a few ways in which one can use the spherical triangle information.
The most direct way would be to perform calculations with the spherical
triangles.  Another way is to use the atom center information to subdivide
the triangles.  One can subdivide a triangle in the molecular surface mesh
into 4, 16, etc. triangles.  The atomic center allows you to place the new
midpoint vertices on the atomic surface.  In this way, the \f$4^n\f$
flat triangles are an approximation of the original spherical triangle.

Note that the initial meshes (the octahedron, the icosahedron, and
their subdivisions) each have less surface area and less volume than
the sphere that they represent.  With a simple transformation, one
could match the surface area, the volume, or some other quantity.
With access to the atomic geometry, one can determine the appropriate
transformation in each case.





<!--------------------------------------------------------------------------->
\section mstAlternatives Alternatives to Triangulations.

One might also represent the surface with a collection of disks.  An oriented
disk in 3-D is defined by a center, a radius, and a normal.  One could
generate a "diskelation" with a similar approach: Diskelate each sphere, and
then clip the disks.
*/


/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page mst_gen Triangulate and Visualize a Generic Molecule

The commands from this usage example are executed from \c stlib/examples/mst.

We generate the triangulation of the generic molecule.  We specify a target
edge length of 1 Angstrom and a probe radius of 0.

\verbatim
./mst.exe -length=0,1 -radius=0 ../../data/mst/gen.xyzr gen.txt \endverbatim

It takes about half a second to generate the surface mesh, which has 28,843
triangles.

Now we visualize the mesh.  We use the
\ref examples_geom_mesh_utility_cellAttributes "cellAttributes"
program to generate cell data.  We generate the modified condition number
(a quality measure) and the content (area) for each of the triangles
in the mesh.  Then we use
\ref examples_geom_mesh_utility_iss2vtk "iss2vtk" to convert the indexed
simplex set representation of the mesh into VTK XML format.  When you open
The VTK file with ParaView, you can visualize the field data on the
triangles.


\verbatim
../geom/mesh/utility/cellAttributes32.exe -mcn gen.txt conditionNumber.txt
../geom/mesh/utility/cellAttributes32.exe -c gen.txt content.txt
../geom/mesh/utility/iss2vtk32.exe -cellData=conditionNumber.txt,content.txt get.txt gen.vtu \endverbatim

Below are three images from ParaView.  I extracted the edges of the mesh
to better detail the triangles.

\image html genA0.0B1.0.jpg "The molecular surface and triangle edges."
\image latex genA0.0B1.0.pdf "The molecular surface and triangle edges."

\image html genA0.0B1.0CN.jpg "The modified condition number."
\image latex genA0.0B1.0CN.pdf "The modified condition number."

\image html genA0.0B1.0C.jpg "The content."
\image latex genA0.0B1.0C.pdf "The content."



We can also triangulate the generic molecule using a function of the atomic
radius as the target edge length.
Below is an example command.  We use the functions \f$0.3 r + 0.5\f$ and
\f$0.6 r + 0\f$.  In the figures below, we visualize the modified condition
number of the triangles.

\verbatim
./mst.exe -length=0.3,0.5 ../../data/mst/gen.xyzr gen.txt \endverbatim

\image html genA0.3B0.5CN.jpg "Edge length function 0.3 r + 0.5 Angstroms.  29,189 triangles."
\image latex genA0.3B0.5CN.pdf "Edge length function 0.3 r + 0.5 Angstroms.  29,189 triangles."

\image html genA0.6B0.0CN.jpg "Edge length function 0.6 r + 0 Angstroms.  19,011 triangles."
\image latex genA0.6B0.0CN.pdf "Edge length function 0.6 r + 0 Angstroms.  19,011 triangles."

*/

} // namespace mst
}

#endif

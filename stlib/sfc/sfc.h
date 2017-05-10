// -*- C++ -*-

#if !defined(__sfc_h__)
#define __sfc_h__

/*!
  \file
  \brief Space-filling curves and orthant tries.
*/

namespace stlib
{
//! All classes and functions in the sfc package are defined in the sfc namespace.
namespace sfc
{

//=============================================================================
//=============================================================================
/*!
\mainpage Space-Filling Curves and Orthant Tries

<!--
CONTINUE: This documentation is horrible. Hopefully I'll have the time to fix 
this later.
-->

\section sfc_tree_data_structures Background: Tree Data Structures

\par
A
<a href="http://en.wikipedia.org/wiki/Tree_(data_structure)">tree</a>
is a hierarchical partition of
objects. Examples include
<a href="http://en.wikipedia.org/wiki/K-d_tree">k-d trees</a>,
which recursively split space in
a single dimension according to the median element. By contrast, in
computational geometry a region tree or 
<a href="http://en.wikipedia.org/wiki/Tree">trie</a>
is a hierarchical partition of the
embedding space. For example, each internal node of a
<a href="http://en.wikipedia.org/wiki/Quadtree#The_region_quadtree">region quadtree</a>
splits space into four congruent quadrants. Tries are also called
<em>digital trees</em>, and in computational geometry may be called
<em>region trees</em>. Generally speaking, the advantage of region 
trees is that the
branching carries implicit geometric information. This may reduce
storage requirements or provide more efficient ways of accessing
nodes. The advantage of trees, is that they can handle difficult
distributions of object locations that are very inhomogeneous or
anisotropic.

\par
<em>Bucket</em> trees store a sequence of objects in each leaf cell. Bucketing
may be applied to any kind of tree, i.e. a k-d tree or
quadtree. Typically one places an upper bound on the number of objects
in a leaf. This is called the bucket size or leaf size. Storing only a
single object at each leaf makes for a simpler implementation, but
using bucketing is almost always more efficient, both in terms of
execution time and storage.

\par
A <em>region quadtree</em> is a region tree, in that it recursively 
divides space. The
top-level cell in the trie spans a finite, axis-aligned rectangle. It
is also called an <em>MX-quadtree</em>, where MX stands for matrix. This is
because the cells at the nth level can represent a
2<sup>n</sup> x 2<sup>n</sup> matrix. A
<em>point quadtree</em> is a tree; the centers of subdivision are always on
points that are stored in the data structure. <em>PR quadtree</em> is an
abbreviation of <em>point region quadtree</em>, and is a region quadtree that
is used to store points. The 3-D equivalent of a quadtree is an
<a href="http://en.wikipedia.org/wiki/Octree">octree</a>.
Dimension-general equivalents are often called hyper-quadtrees
or n-dimensional quadtrees. The author prefers the term <em>orthant trie</em>
for the dimension-general quadtree.

\par
Quadtrees traditionally use the same branching structures as other
trees, like the k-d tree. Internal nodes store pointers to
children.
\ref sfc_gargantini_1982 "Gargartini (1982)".
introduced the <em>linear quadtree</em>,
which uses
<a href="http://en.wikipedia.org/wiki/Z-order_curve">Morton codes</a>
to order the cells. These
codes are formed by interleaving the bits of the index coordinates of
cells. The cells may be stored in a contiguous block of memory. The
Morton order improves the locality of reference for algorithms that
operate on the trie and hence improves cache utilization. Furthermore,
linear quadtrees use less storage than implementations that use
pointers. Finally, they are more suitable for concurrent data
structures. It is fairly easy to adapt a serial linear quadtree to a
distributed memory one. Depending upon the application, the cells may
be partitioned among the processes, or they may overlap.



\section sfc_space_filling_curves Space-Filling Curves

\par
<a href="http://en.wikipedia.org/wiki/Space-filling_curve">
Space-filling curves</a> are of interest in a variety of fields. They are of
interest here because they may be used to order the indices of a
multi-dimensional array. Specifically, they are used to order the indices
of an array whose extents are 2<sup>n</sup> in each dimension.
\ref sfc_samet_2006 "Samet (2006)".
discusses various space-filling curves and their properties
in Section 2.1.1.2, "Ordering Space."
As we will apply them to trie data structures, the ones that are relevant
for us are recursive: i.e.
<a href="http://en.wikipedia.org/wiki/Z-order_curve">Morton order</a>,
<a href="http://en.wikipedia.org/wiki/Hilbert_curve">Peano-Hilbert order</a>,
Gray order, double Gray order, and U order.

\par
Even if you have never heard the term "space-filling curve," you are 
probably somewhat familiar with the concept. If you know how multi-dimensional
array indices are converted to a single index for accessing elements 
in storage, then you have encountered a space-filling curve. Consider 
a 3-D array with extents <em>N</em> in each dimension. There are a total
of <em>N<sup>3</sup></em> elements in the array. If the the array has
column-major storage order, then the element at position <em>(i, j, k)</em>
is stored at the offset <em>n = i + jN + kN<sup>2</sup></em>. 
The mapping from multi-indices to the storage index defines a space-filling
curve. In following the storage index from 0 to <em>N<sup>3</sup>-1</em>, 
each element of the array is visited once.

\par
Column-major storage order has the advantage that it is easy to compute. 
One can convert a cell location to a storage index with a few additions and 
multiplications. Converting back to the multi-index simply requires  
integer division and addition. The situation is even easier when the
array extents are powers of 2. Then you can avoid integer multiplications 
and divisions entirely and just use bit shifting. The disadvantage of
column-major storage order is that it often leads to mediocre locality of
reference. Consider a calculation in which you access the six adjacent cells
of a cell. The two adjacent cells in the first dimension are ideally located,
they are adjacent in storage. However, the adjacent cells in the second
dimension are at offsets of <em>N</em>, while for the third dimension
the offsets are <em>N<sup>2</sup></em>. For large arrays, these are 
far away in storage.

\par
When people mention space-filling curves, they usually are referring to 
Morton, Hilbert, or the like, that is, curves that have a recursive 
structure, and as a result enable algorithms with good locality of 
reference. But don't be intimidated, you can just think of these curves
as a fancy way of indexing arrays. Because of their elegant mathematical
structure, they have some advantages and capabilities that simpler orderings,
like column-major, do not.



\section sfc_codes_and_blocks Codes and Blocks

\par Structure of codes.
Each of the orders (Morton, Hilbert, Gray, double Gray, and U) maps
index coordinates with extents of [0 .. 2<sup>n</sup>) in each dimension
into integers in the range [0 .. 2<sup><em>Dn</em></sup>). This indexing is
interpreted as a location code. At the first level of refinement,
<em>n = 1</em>, the domain is divided into 2<sup><em>D</em></sup> orthants and
locations are encoded in <em>D</em> bits. For <em>n</em> levels of refinement,
the directions for each level are recorded in <em>D</em> bits. The first
level of refinement uses the <em>D</em> most significant of the
<em>Dn</em> bits, while the last level uses the least significant bits.
The Morton code is the simplest to construct; one simply interleaves the
bits of the indices.

\par Guard code.
To represent a code, we use an unsigned integer type with at least
<em>Dn + 1</em> bits. The extra bit is so that we can represent a integer
that is greater than any valid code. Specifically,
this guard code is the largest integer, i.e. every binary digit of the
unsigned integer is 1.
When codes represent cells in a trie,
we always terminate the sequence of codes with the guard code in order to
simplify checking for the end of the sequence. However, when codes record
locations for objects (with a one-to-one mapping between objects and codes) 
the resulting sequence
of codes is not terminated with the guard code.
(Apologies for jumping ahead a bit, we'll talk about tries later on.)

\par Blocks.
A location code is a representation of the indices in a multidimensional array.
If one considers an array of cells that covers a Cartesian domain, then
it represents the location of the lower corner of a cell.
If all of the cells are at the same
level of refinement,
location codes alone are sufficient to identify cells.
While we have data structures for a single level of refinement, we 
also have data structures that have cells
at multiple levels. Since cells at different levels may have the same
lower corner location, we will also need to record the cell
level.
Using codes
with <em>n</em> levels of refinement, one can represent a cell at
level <em>k</em> by only using the most significant <em>Dk</em> bits
and by recording the level. This combination of a (restricted) location
and the level is called a <em>block</em>. A block at level <em>k</em>
spans 2<sup>n-k</sup> cells in the fully refined array.
There is a single block at level 0 that spans the entire domain;
blocks at the highest level span a single cell.

\par
Some store the location code and level as a tuple, but we store
them in a single integer, as does
\ref aboulnaga_2001 "Aboulnaga et al (2001)".
They store
the level in the least significant bits and the Morton code in the
most significant bits. The resulting location and level represents a
Morton block. We follow the same approach.
Specifically, we represent a block by left-shifting the location code
and recording the level in the least significant bits.
Thus, blocks are ordered first by the location of their lower corner
and then by level. As an additional feature,
we reserve
one of the most significant bits to represent a cell that lies past
all valid cells. Adding a guard cell with this block value to the end of
the cell list makes some iteration procedures more efficient.
Specifically, the guard cell frees us from explicitly checking to see
if we have reached the end of the list of cells.


\par Data structures.
When using space-filling curves, there are a few fundamental traits
that are common each of the resulting data structures. We record these
in the Traits class. In its template parameters one specifies the
following:
- The space dimension.
- The floating-point number type.
- The unsigned integer type used for codes.
- The type of the space-filling curve, for example, Morton.
.
For the sake of simplicity and consistency, most of the classes in this 
package take the Traits class as their first template parameter.

\par
You will probably not directly use the classes for calculating codes and blocks.
However, we point out their implementations here for those that are
interested in what is going on under the hood.
We use the LocationCode class to calculate location codes from Cartesian points.
The BlockCode class calculates blocks. It has the same template parameters.

\par
We use the MortonOrder class to calculate Morton codes. This class may be
used as the final template parameter for Traits.
Implementations for other orderings, such as Hilbert, may be added later.
Calculating Morton block codes, as well as converting the codes
back to integer coordinates, can be a costly operation in some applications.
\ref stocco_2009 "Stocco and Schrack (2009)"
present efficient methods for interleaving
bits. Specifically, they use shifts along with bitwise and’s and or’s
to dilate and contract binary integers. The number of operations for
an <em>n</em>-bit integer is log<sub>2</sub><em>n</em>.
\ref raman_2008 "Raman and Wise (2008)" present both table
lookup methods and arithmetic methods.
In MortonOrder we use a table lookup method. If the cost of generating
codes proves costly, we will examine other methods.




\section sfc_bucketing_and_tries Bucketing and Tries

\par Object requirements.
This package implements several related data structures for performing
spatial queries on a set of objects. 
If objects have associated point locations,
then we can use a space-filling curve to order them.
Specifically, the location is used to assign objects to cells.
We use the centroid of the tight axis-aligned bounding box for the object
as the location. That is, 
for an object \c x, we compute \c centroid(geom::bbox<BBox>(x)).
Thus, for any object type that is used with the SFC cell data structures,
building the bounding box must be supported. (See geom::BBox for details.)

The SFC data structures dictate an
ordering for the objects, however, they do not store them - it is up to the
user to do that. If the object type is implemented as a self-contained class,
then one might simply store them in a \c std::vector. For a more complicated
example, the objects might be elements in a simplicial mesh,
perhaps implemented with \c geom::mesh::IndexedSimplexSet.
Regardless of how objects are represented and stored, we require that they
can be reordered according to the space-filling curve.
Also, one must be able to access objects by the resulting index.

\par Cells.
The SFC data structures each store an array of cells.
The cells are ordered by location codes or block codes.
These codes are stored in a separate array. 
The most commonly stored information for cells is a representation of the
objects indices. Since the objects are ordered according to the cell
to which they belong, there is no need to store each of the indices.
Each cell holds a contiguous range of objects, so an index range
defines the contained objects.
One could store a standard index range in each cell,
that is, one could store the first index and one past the last index.
However, this approach would use more storage than is necessary.
It is more efficient to only store the first index. The end of
the range of objects indices is equal to the first index in the next cell.
(Actually, the definition of \e next depends upon whether the cells overlap, 
but we won't worry about that detail for now. Just think about
non-overlapping cells.) Since storing object indices is so common,
each of the SFC data structure has a template parameter that specifies
whether this information should be stored.

\par
Each of the SFC data structures have a template parameter for specifying 
the cell type. A cell holds
information about the objects that are associated with it. (Note that since the
objects may have nonzero extent, they may not be contained in the cell.)
For example, it is common to
store a tight bounding box for the objects. For this, one would use
a cell type of geom::BBox. There are
certain requirements for classes that implement a cell. We will
consider these requirements in detail in the
\ref sfc_cell_properties section. For now, it is
sufficient to know that cells hold information about their
associated objects.

\par Quasi-static.
There are many ways of using the ordering of space-filling curves
to implement data structures that support queries.
Cells could be stored in a B-tree or
simple sequence. The way that you implement the data structure depends on
what you want to do with it. The one thing that they all have in
common is that they use codes to represent the cells. (Specifically,
they either use a location code to represent the lower corner of a cell
or they use a block code to represent the lower corner and the level 
of a cell.) Some applications
require dynamic trees that allow efficient insertion and deletions of
elements. For example,
\ref aboulnaga_2001 "Aboulnaga et al (2001)"
store quadtree blocks in
a B<sup>+</sup>-tree. Insertions, deletions, and lookups can all be accomplished
with amortized logarithmic computational complexity.
By contrast, the data structures in this package are specialized for
quasi-static problems. The codes and cells are stored in stored in
simple arrays (implemented with \c std::vector). Most often, the objects
would also be stored in a contiguous block of memory. This representation
enables very efficient implementations of bulk operations. Because of
the ordering from the space-filling curve and layout in memory,
such operations have good locality of reference, and thus have good
cache utilization.
The data structures in this package are useful in applications in which
the operations may be applied in bulk. Operations such as building, coarsening,
and merging have linear asymptotic computational complexity in
the number of objects and/or the number of cells.
Because operations such as inserting
or deleting a single object would not be efficient, they are not implemented.

\par Ordered cells.
As we mentioned before, each of the SFC data structures stores an array
of the cells and an array of their associated codes. They each derive from
the OrderedCells class, which implements the common functionality.
You will not directly construct
this class, but you will use its member functions.
For example, you can access information about the Cartesian domain with
OrderedCells::lowerCorner(), and OrderedCells::lengths().
OrderedCells::numLevels() returns the number of levels of refinement.
You can access cells with OrderedCells::operator[] and
OrderedCells::size().

\par Uniform bucketing.
The simplest query data structure in this package is UniformCells.
As the name suggests, the cells are of uniform size. One can specify
the domain and the number of levels of refinement in the constructor.
Alternatively, one can construct it using a bounding box for the
object locations and a minimum acceptable cell length. In any case,
cells are only created at the finest level of refinement.
Thus, the level is not encoded in the location codes.
The UniformCells class is useful for applications in which it is
useful to apply bucketing to the objects at a specific length scale.
Since all of the cells are at the same level, the operations of
coarsening (reducing the levels of refinement) and merging are
simple and efficient.

\par Multi-level bucketing.
The AdaptiveCells class stores cells that may be at different levels
of refinement. Thus, it uses the BlockCode class for ordering. The
cells are non-overlapping, but do not necessarily cover the whole
domain because empty cells are not stored. This class is useful when
the bucketing that one desires cannot be defined in terms of a single
length. For example, one could generate cells that are as large as
possible, but do not exceed a certain number of objects. Note that
when one calls OrderedCells::build() to order the objects and insert
cells, the cells are created at the highest level of refinement. One
can then use one of the coarsening functions to generate cells with
the desired properties. For example, one could use
AdaptiveCells::coarsenCellSize() to generate cells with a specified
maximum number of objects.

\par Linear orthant trie.
The LinearOrthantTrie class implements a linear orthant trie, which
like AdaptiveCells has non-overlapping leaf cells at multiple levels,
but which also has internal cells. Specifically, every cell except
for the root cell (at level 0), has a parent cell in the trie.
Thus, one may reach any cell by following a path from the root.
For traversing the trie from the top down, we need to access children of
cells. We could store the indices of the children in each cell, but
this requires 2<sup>D</sup> indices per cell. Instead, we store the index of the
next cell on the same level. (This is implemented by storing an array
of the next indices.) The ordering from the space-filling curve enables
this compact representation.
Note that if a cell is not a leaf, then
its first child immediately follows it in the sequence of sorted
cells. Thus, we can directly access the first child. We access the rest
with the indices of the next cells.

\par
As with the other query data structures, you can construct a
LinearOrthantTrie by specifying the domain in the constructor
and then calling LinearOrthantTrie::build() with the vector of
objects. Alternatively, you can construct one from a
AdaptiveCells. This is useful when you want to add internal cells
to a set of non-overlapping cells. For example, one could create a
AdaptiveCells data structure, apply coarsening to acheive the desired
bucketing, and then construct a LinearOrthantTrie from the result to
obtain a trie that may be traversed in top-down fashion.
For traversing the trie, use the accessors:
LinearOrthantTrie::isLeaf(),
LinearOrthantTrie::isInternal(), and
LinearOrthantTrie::getChildren().



\section sfc_cell_properties Cell Properties

\par Generic cells.
For UniformCells and AdaptiveCells,
you can use any data type as the cell type (as long as it is default 
constructable and copy constructable). However, with a generic cell type
only limited functionality is available. For example, you can't construct
the cell data structures from a vector of objects, because a generic 
cell data type can't be constructed from a range of objects.
To access this functionality, the BuildCell functor must be specialized
for the cell type. There are already specialization for 
geom::BBox and geom::ExtremePoints in Cell.h. If you you want to 
use a new cell type, follow these implementations.
If a cell type does not have a BuildCell specialization,
you must build the cell data structure from 
a vector of code/cell pairs. Specifically, use the 
OrderedCells::build(std::vector<std::pair<Code, Cell> > const&)
function.

In order to coarsen or otherwise merge cells in an SFC data structure,
there must be a method for adding one cell to another.
Such operations may only be used if the += operator has been 
defined for the cell type. Note that internal cells in LinearOrthantTrie
are built by coarsening groups of sibling cells. Thus, one may 
only use LinearOrthantTrie with cells for with the += operator is defined.


\section sfc_bibliography Bibliography

-# \anchor sfc_gargantini_1982
Gargantini, Irene. "An effective way to represent quadtrees."
Communications of the ACM 25.12 (1982): 905-910.
-# \anchor sfc_samet_2006
Samet, H. (2006). Foundations of multidimensional and metric data structures.
Amsterdam: Elsevier/Morgan Kaufmann.
-# \anchor aboulnaga_2001
Aboulnaga, Ashraf, and Walid G Aref.
"Window query processing in linear quadtrees."
Distributed and Parallel Databases 10.2 (2001): 111-126.
-# \anchor stocco_2009
Stocco, Leo J, and Gunther Schrack. "On spatial orders and location codes."
Computers, IEEE Transactions on 58.3 (2009): 424-432.
-# \anchor raman_2008
Raman, Rajeev, and David S Wise. "Converting to and from dilated integers."
Computers, IEEE Transactions on 57.4 (2008): 567-573.

*/

}
}

#endif

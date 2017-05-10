// -*- C++ -*-

#if !defined(__amr_h__)
#define __amr_h__

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/Traits.h"

/*!
  \file amr.h
  \brief Includes the adaptive mesh refinement classes.
*/

namespace stlib
{
//! All classes and functions in the amr package are defined in the amr namespace.
namespace amr {}
}

//=============================================================================
//=============================================================================
/*!
  \mainpage Adaptive Mesh Refinement

  \par
  This package is a C++ templated class library designed to support adaptive
  mesh refinement (AMR) for
  logically Cartesian, structured meshes. A heirarchy of sub-grids are stored
  in a tree data structure. Each node of the tree is a structured mesh with
  the same index ranges. In N-dimensional space, refinement replaces a single
  mesh with 2<sup>N</sup> meshes; coarsening does the opposite.

  \par
  Notable features include:
  - Support for any space dimension. (The dimension is a template parameter.)
  - Mixed-mode concurrency (both shared-memory and distributed-memory) to
  effectively utilize a wide range of architectures, including clusters
  of multi-core processors.

  \par
  This package has a <em>linear orthree</em> called amr::Orthtree .
  Each leaf has an associated <em>spatial index</em> (implemented in
  amr::SpatialIndexMorton) that encodes its level and position in the tree.
  For distributed-memory applications, use amr::DistributedOrthtree .

  \par
  Use this sub-package by including the desired class or by including
  the file amr/amr.h.

  <!------------------------------------------------------------------------>
  \section amr_introduction Introduction

  \par
  A tree data structure that recursively divides a 2-D region into quadrants
  is called a \e quadtree.  (http://en.wikipedia.org/wiki/Quadtree)
  The 3-D equivalent is
  an \e octree.  The N-D analogue of the 2-D quadrant and the 3-D octant
  is an \e orthant.  (http://en.wikipedia.org/wiki/Orthant)
  In N-D space there are 2<sup>N</sup> orthants.  A tree that
  recursively divides space into orthants has been called an
  <em>N-D quadtree</em> and a <em>hyper-octree</em>.  <!--CONTINUE cite-->
  We think the term \e orthtree is a descriptive and succinct name for such
  data structures.  As we have implemented a data structure that is generic
  in the space dimension (the dimension is a template parameter) we will
  use the term orthtree instead of quadtree or octree.

  \par
  There are two primary methods of representing an orthtree:
  <em>pointer-based</em> and \e linear. With a pointer-based representation,
  one stores the branches as well as the leaves of the tree.
  Here we use the term \e node to refer to either a branch or a leaf.
  Each branch stores 2<sup>N</sup> pointers to nodes in the tree.
  These pointers to the \e children of each branch are necessary for
  searching for a leaf.  Usually, each node also stores a pointer to its
  \e parent.  This is convenient for many algorithms, including traversing
  the nodes.  The position of a leaf is implicitly stored in the branches
  leading from the root to the leaf.

  \par
  With the linear representation of an orthtree, only the leaves are stored.
  Instead of implicitly representing the positions of the leaves with branches,
  <em>spatial indices</em> are used to explicitly describe these positions.
  In this context, we call the pair of a spatial index and a leaf a \e node.
  "Linear" refers to the fact that the nodes may be stored in a sequence
  container (an array, for example).  A specific node may be accessed
  by searching for its spatial index.

  <!------------------------------------------------------------------------>
\section amr_considerations Considerations

\par
There are a few basic design issues to consider when using AMR. First consider
how one refines a region. There are several possibilities:
- Finer grids are formed from the orthants of a single coarser grid.
This leads to a regular tree data structure.
- Finer grids are arbitrary rectilinear portions of a single coarser grid.
This leads to an irregular tree data structure.
- Finer grids may be arbitrary rectilinear portions of one or more coarser
grids. In this case the hierarchy of grids could be represented with a graph.
.
We will name these three refinement schemes \e orthant, \e nested, and
\e arbitrary.

  <!------------------------------------------------------------------------>
\section amr_orthant Orthant Arithmetic


\subsection amr_orthant_numbering Orthant Numbering

\par
The 2<sup>N</sup> orthants are numbered from 0 to 2<sup>N</sup> - 1.
The following figure and table show this numbering for the 2-D case of
quadrants.

\image html quadrant.jpg "The quadrants."

\par
<table>
<tr> <th> Number <th> y <th> x <th> Binary Number
<tr> <td> 0 <td> - <td> - <td> 00
<tr> <td> 1 <td> - <td> + <td> 01
<tr> <td> 2 <td> + <td> - <td> 10
<tr> <td> 3 <td> + <td> + <td> 11
</table>

\par
Note that the origin is in the lower left corner.  In computer graphics, one
often places the origin in the \e upper left corner so that row numbering
starts at the top of the screen.  We instead use the convention that is
more familiar in geometry and physics.

\par
The table below lists octant numbering.
<table>
<tr> <th> Number <th> z <th> y <th> x <th> Binary Number
<tr> <td> 0 <td> - <td> - <td> - <td> 000
<tr> <td> 1 <td> - <td> - <td> + <td> 001
<tr> <td> 2 <td> - <td> + <td> - <td> 010
<tr> <td> 3 <td> - <td> + <td> + <td> 011
<tr> <td> 4 <td> + <td> - <td> - <td> 100
<tr> <td> 5 <td> + <td> - <td> + <td> 101
<tr> <td> 6 <td> + <td> + <td> - <td> 110
<tr> <td> 7 <td> + <td> + <td> + <td> 111
</table>

\par
This is a nifty numbering system because the binary representation of
the orthant encodes the \e N coordinate directions (0 for negative and 1 for
positive).  Specifically, the nth bit (starting with the
least significant bit) encodes the direction for the nth
coordinate.  Let the coordinates be labeled from 0 to \e N - 1.
If \e i is the orthant number, then we can get the direction bit (0 or 1) for
the nth coordinate with a shift and a modulus.
\code
(i >> n) % 2 \endcode
It easy to transform this to a direction sign (\f$\pm 1\f$).
\code
((i >> n) % 2) * 2 - 1 \endcode
By the way, you can express multiplication and division by a power
of 2 as left or right shifting.  <code>x * 2</code> is the same as
<code>x << 1</code>. Any reasonable compiler will generate the same
(efficient) code regardless of whether you use the former or latter method.


\subsection amr_orthant_directions Directions

\par
In N-D space there are 2N signed coordinate directions.  We label
these with integers from 0 to 2N - 1.  In the figure and table below
we show the direction numbering in 2-D.

\image html directions.jpg "The direction numbering in 2-D."

\par
<table>
<tr> <th> Number <th> Direction
<tr> <td> 0 <td> -x
<tr> <td> 1 <td> +x
<tr> <td> 2 <td> -y
<tr> <td> 3 <td> +y
</table>

\par
The next table gives the direction numbering in 3-D.
<table>
<tr> <th> Number <th> Direction
<tr> <td> 0 <td> -x
<tr> <td> 1 <td> +x
<tr> <td> 2 <td> -y
<tr> <td> 3 <td> +y
<tr> <td> 4 <td> -z
<tr> <td> 5 <td> +z
</table>

\par
This is a nifty numbering scheme because is easy to extract the coordinate
number and the coordinate direction.  For direction <code>i</code>
the former is <code>i / 2</code> and the latter is <code>i % 2</code>.
(Here we use 0 and 1 to indicate negative and positive.)


  <!------------------------------------------------------------------------>
\section amr_index The Spatial Index

\par
The geometry of a node in an orthtree may be described by its level and
the \e N index coordinates of its lower corner.  There are various ways of
storing this information.  One can use an integer to store the level and
\e N integers to store the index coordinates.  Alternatively one can pack
this information into an integer or bit array.  For the packed case, one
can store each index coordinate in contiguous bits, or one can interleave
the bits.  Typically, the method of storage directly translates into
an ordering of the nodes.  If one packs the information into a single
integer type, then one uses the standard less than comparison on
integers to order the nodes.

\par
There is a standard spatial indexing scheme called the \e quadcode which
stores the
level and coordinates in a single integer type.  The index coordinates
are interleaved and packed into the most significant bits.  The level is
stored in the less significant bits.  This translates into an ordering of
the nodes that has some handy properties.  Most importantly, there is
a spatial coherence to a block of nodes accessed in order.  This is useful
both from an algorithmic perspective (some searching operations are easier)
and for performance (fewer cache misses).  In addition, this ordering can
be used for partitioning the nodes across processors.

\image html zOrdering.jpg "Z-ordering."

\par
While the quadcode is handy for ordering the nodes, a non-interleaved
representation is more useful for manipulating the spatial index.  When
determining the parent, children, or neighbors of a node, one needs to
manipulate the index coordinates and level.  With the quadcode, one extracts
the coordinates and level, manipulates them, and then packs them back
into the quadcode.  Because the bits are interleaved, this unpacking and
packing is fairly expensive.  (The computational complexity of both packing
and unpacking is linear in the depth of the tree.)  If one uses a
separated or non-interleaved representation for the spatial index, then
manipulations can be done in constant time.

\par
For our implementation, we store the interleaved quadcode in a single
integer type and also separately store the index coordinates and level.
We use the quadcode for ordering the nodes and use the separate
index coordinates and level for spatial index arithmetic.  This makes
the algorithms easier to implement and more efficient, but
roughly doubles the storage requirements for the spatial index.
If one is storing a significant amount of data in each node (a small
grid, or a spectral element), then this extra storage requirement for the
spatial indices will be negligible.


  <!------------------------------------------------------------------------>
\section amr_container The Node Container

\par
Each node in the orthtree has a spatial index and data.  For a static tree,
one can store the nodes in an array, sorted by the spatial indices.  One
can then find a node in logarithmic time with a binary search.  For a dynamic
tree, storing the nodes in an array may not be efficient because inserting or
deleting a node has linear complexity in the size of the array.  One can
mitigate this by caching the insertions and deletions.  Another drawback
to storing the nodes in an array is that insertions and deletions invalidate
pointers.  If one were storing iterators to adjacent neighbors, this
information would need to be recomputed following an insertion or deletion.
A different approach (and the one that we have taken) is to store the nodes
in a paired associative container. <!--CONTINUE cite--> We use
\c std::map .  The node is a \c std::pair with \c first being
a const spatial index and \c second being the data.  A node can still be
found in logarithmic time.  Inserting new nodes does not invalidate any
iterators.  Deleting a node only invalidates iterators to the deleted
node.  Finally, when iterating through the nodes, they are ordered according
to the spatial index.


<!-------------------------------------------------------------------------->
\section amr_concurrency Concurrency

\subsection amr_concurrency_distributed Distributed Memory

\par
The strategy for achieving distributed-memory concurrency is to partition the
nodes and distribute them across processors.  Each processor has an orthtree
that holds a subset of the nodes.  The domain of each orthtree is the same.
The nodes stored in a particular orthtree cover a portion of its domain.

\par
There are many ways to partition the nodes across processors.  The simplest
method is to use the z-ordering and give each processor a contiguous block
of nodes.  The number of nodes in each processor is approximately equal.

\subsection amr_concurrency_shared Shared Memory

\par
The strategy for achieving shared-memory concurrency is to partition a
processor's nodes across threads.  Let there be \e N threads available in
concurrent sections.  When applying an operation to all nodes, the set of
nodes is partitioned into \e N subsets using the z-ordering.  Then each thread
acts on its set of nodes.  Here we assume that an operation applied to a node
may alter only that node.  However, it may access the state of other nodes.
We also assume that the operations may be applied in any order.  The threads
may concurrenty alter the state of their nodes without conflict.

\par
Refinement is more complicated than applying an operation to the nodes
because it involves inserting and erasing nodes.  One could make
insertions and deletions thread-safe by performing these operations in
OpenMP critical sections.  (Recall that only one thread may execute a
given critical section at a time.)  This appoach is easy to implement,
but may not be efficient because of its fine-grained nature.  It is
usually preferable for threads to cache their modifications and
perform them in a single critical section.  Note that determining
which nodes should be split may be done concurrently, each thread
examines its subset of the nodes.  After each thread collects the
nodes that need refinement, it applies splitting operations to those
nodes in a critical section.

\par
The concurrent algorithm for coarsening is similar to that for refinement.
However, the partitioning must be done differently.  Coarsening is done
by repeating sweeps until no further coarsening is required.
In a coarsening sweep, a group of nodes may be merged a single time.
Before each sweep, the nodes are partitioned such that each potential
merging operation involves nodes in only one of the subsets.  The threads
concurrently collect the groups of nodes that need coarsening.  Then
each thread performs the merges in a critical section.

\par
The figure below shows two partitions of a 2-D orthtree.  The first partition
evenly divides the nodes and is suitable for applying an operation to the
node data or for refinement.  It is not suitable for coarsening because
a mergeable group of nodes is divided between the two sets.  The second
partition can be used for a concurrent coarsening sweep.  Each set contains one
group of nodes that could be merged.

\image html partition.jpg "Two partitions of an orthtree."

  <!------------------------------------------------------------------------>
\section amr_performance Performance

\par
For timing, we use a MacBook Pro with a 2.8 GHz Intel Core 2 Duo processor,
4 GB of 1067 MHz DDR3 RAM, and 6 MB of L2 cache.

\subsection amr_performance_AssociativeContainers Associative Containers

\par
As linear orthtrees use associative containers to store the leaves of the
tree, we first consider the performance of these. We consider three containers:
\c std::map, \c __gnu_cxx::hash_map, and \c std::unordered_map.
\c std::map uses a tree data structure; the computational complexity
of insertions, deletions, and accessing elements is logarithmic in the
number of elements. The latter two use hashing and have constant computational
complexity for these operations.

\par
The container holds N elements which are pairs of
<code>std::size_t</code>. The key and the value are the same and are the
integers in the range [0..N).
We measure the time it
takes to find an element. Specifically, we use the \c count member function
to verify that an element exists. For these searching operations,
the keys are shuffled.

\par
The source code for the timing programs is in \c test/performance/stl.
Use SCons to build them.
\verbatim
cd test/performance
scons stl
\endverbatim

\par
The scripts that run the programs are in \c results/amr/AssociativeContainers.
The tables below give the search times is nanoseconds. Use "make export"
to regenerate the results.
\verbatim
cd results/amr/AssociativeContainers
make export
\endverbatim

\par
\htmlinclude AssociativeContainersMap.txt
Performance for <code>std::map</code>.

\par
\htmlinclude AssociativeContainersHashMap.txt
Performance for <code>__gnu_cxx::hash_map</code>.

\par
\htmlinclude AssociativeContainersUnorderedMap.txt
Performance for <code>std::unordered_map</code>.

\par
We see that for \c std::map the search times depend on the number of
elements. There is a steep increase in costs in going from 100,000
elements to 1,000,000 elements. This is because the data for the
former problem will fit in the L2 cache, while the data for the latter
will not. Cache misses become the dominant cost.

\par
The two data structures that use hashing have identical performance
and are significantly faster than \c std::map. Again, the performance
takes a serious hit when the problem no longer fits in the L2
cache. Although the associative containers that use hashing are faster
\c std::map, the latter has the advantage that the elements are
sorted. This makes partitioning easier, for example. Thus the orthtree
uses \c std::map to store leaves.


\subsection amr_performance_SpatialIndexMorton Spatial Index Performance

\par
We consider the performance of various operations on the spatial index:
- \b Set - Set the level and coordinates.
- \b Parent - Transform a spatial index to its parent.
- \b Child - Transform a spatial index to one of its children.
- \b Next - Transform a spatial index to the next index.
- \b Neighbor - Transform a spatial index to an adjacent neighbor.
.
These are the operations that are used in refinement, coarsening, and finding neighbors.

\par
The source code for the timing programs is in
\c test/performance/amr/SpatialIndexMorton. Use SCons to build them.
\verbatim
cd test/performance
scons amr/SpatialIndexMorton
\endverbatim

\par
The script that runs the programs is in \c results/amr/SpatialIndexMorton.
\verbatim
cd results/amr/SpatialIndexMorton
make export
\endverbatim

\par
\htmlinclude SpatialIndexMorton3.txt
Execution time in nanoseconds for a spatial index in 3-D.

\par
As expected, the execution time increases with the number of levels.
However, the timings for 16 levels is higher than those for 20 levels.
\todo Find out why.



\subsection amr_performance_serial Serial Performance

\par
We will consider the performance of various operations on the nodes of
an orthtree. The source code for the timing programs is in
\c test/performance/amr/Orthtree. Use SCons to build them.
\verbatim
cd test/performance
scons amr/Orthtree
\endverbatim

\par
The scripts that run the programs are in \c results/amr/Orthtree.
\verbatim
cd results/amr/Orthtree/serial
make export
\endverbatim


\subsubsection amr_performance_serial_access Node Access

\par
First we consider the cost of accessing nodes in the tree. There are two
ways of accessing nodes:
-# ordered access - Iterate over the nodes from beginning to end.
-# random access - Find a specified node, which is accomplished with a
binary search.
.
We perform tests on an orthtree with 2<sup>12</sup> = 4096 nodes.

\par
First consider ordered access of the nodes. This test is performed
on an orthtree with 2<sup>12</sup> = 4096 nodes.
The time per access is given in nanoseconds.

\par
\htmlinclude OrthtreeAccessOrdered.txt
Ordered access in a range of dimensions.

\par
\htmlinclude OrthtreeAccessOrdered3.txt
Ordered access in a 3-D orthtree of varying sizes.

\par
Next we consider random access of the nodes.  Again this test is
performed on an orthtree with 2<sup>12</sup> = 4096 nodes.

\par
\htmlinclude OrthtreeAccessRandom.txt
Random access in a range of dimensions.

\par
\htmlinclude OrthtreeAccessRandom3.txt
Ordered access in a 3-D orthtree of varying sizes.

\par
We see that ordered access is much faster than random access.
The dimension of the orthtree has little effect on the performance.
The access time increases with the number of nodes. For large trees,
cache misses become the dominant cost.



\subsection amr_performance_serial_refinement Refinement

\par
\htmlinclude OrthtreeRefine.txt
Refinement for a range of dimensions. The final orthtree has
2<sup>12</sup> = 4096 nodes.

\par
\htmlinclude OrthtreeRefine3.txt
Refinement for a 3-D orthtree of varying final sizes.

\par
The cost of refinement increases with increasing dimension. In higher
dimensions, a refined node has more children. There is a weak dependence
on the number of nodes in the tree; the dominant cost is in generating
the children. However, there is a significant cache-miss penalty for
larger trees.



\subsection amr_performance_serial_coarsening Coarsening

\par
\htmlinclude OrthtreeCoarsen.txt
Coarsening for a range of dimensions. The initial orthtree has
2<sup>12</sup> = 4096 nodes.

\par
\htmlinclude OrthtreeCoarsen3.txt
Coarsening for a 3-D orthtree of varying initial sizes.

\par
The cost of coarsening is similar to the cost of refinement.



\subsection amr_performance_serial_neighbors Finding Neighbors

\par
\htmlinclude OrthtreeNeighbors.txt
Finding neighbors in orthtree of varying dimensions.
This test is performed on orthtrees with
2<sup>12</sup> = 4096 nodes.

\par
\htmlinclude OrthtreeNeighbors3.txt
Finding neighbors for a 3-D orthtree of varying sizes.

\par
The cost of finding neighbors increases with dimension because in N-D
space there are 2N adjacent neighbors (if the neighbors are all at the
same level of refinement).



  <!------------------------------------------------------------------------>
\section amr_compiling Compiling

\par
To use this package, you will need a compiler that supports the C++ TR1
library. GCC 4.2 and beyond should work.


  <!------------------------------------------------------------------------>
  \section amr_examples Examples

  \par
  There are a number of examples in \c stlib/examples/amr that show how to
  use the orthtree:
  - Use refinement to \ref examples_amr_distance "capture the zero iso-surface of a ball".
  - \ref examples_amr_synchronize "Synchronize" data between adjacent patches.


  <!------------------------------------------------------------------------>
\section amr_links Links

\par Information.
- http://en.wikipedia.org/wiki/Quadtree
- Visit http://donar.umiacs.umd.edu/quadtree/index.html for spatial index
demos using Java.

\par Software.
- Chombo http://seesar.lbl.gov/anag/index.html
- Overture https://computation.llnl.gov/casc/Overture/
- HyperCLaw http://seesar.lbl.gov/ccse/Software/index.html
*/

#endif

// -*- C++ -*-

#if !defined(__shortest_paths_h__)
#define __shortest_paths_h__

#include "stlib/shortest_paths/GraphBellmanFord.h"
#include "stlib/shortest_paths/GraphDijkstra.h"
#include "stlib/shortest_paths/GraphMCC.h"
#include "stlib/shortest_paths/GraphMCCSimple.h"

//=============================================================================
//=============================================================================
/*!
\mainpage Single-Source Shortest Paths

- \ref shortest_paths_introduction
- \ref shortest_paths_dijkstra
- \ref shortest_paths_greedier
- \ref shortest_paths_complexity
- \ref shortest_paths_performance
- \ref shortest_paths_concurrency
- \ref shortest_paths_future
- \ref shortest_paths_conclusions
*/


//=============================================================================
//=============================================================================
/*!
\page shortest_paths_introduction Introduction



Consider a weighted, directed graph with \e V vertices and \e E edges.
Let \e w be the weight function, which maps edges to real numbers.  The
weight of a path in the graph is the sum of the weights of the edges
in the path.  Given a vertex \e u, some subset of the vertices can be
reached by following paths from \e u.  The <em>shortest-path weight</em>
from \e u to \e v is the minimum weight path over all paths from
\e u to \e v.  A <em>shortest path</em> from
vertex \e u to vertex \e v is any path that has the minimum weight.
Thus the shortest path is not necessarily unique.  If there is no path
from \e u to \e v then one can denote this by defining the shortest-path
weight to be infinite.


We consider the <em>single-source shortest-paths problem</em>
cite{cormen:2001}: given
a graph and a <em>source</em> vertex we want to find the shortest paths
to all other vertices.   There are several related shortest-paths problems.
For the <em>single-destination shortest-paths problem</em>
we want to
find the shortest paths from all vertices to a given <em>destination</em>
vertex.  This problem is equivalent to the single-source problem.  Just
reverse the orientation of the edges.  For the
<em>single-pair shortest-path problem</em>
we want to find the shortest path from a given
source vertex to a given destination vertex.  One can solve the single-pair
problem by solving the single-source problem.  Actually, there are no known
single-pair algorithms with lower computational complexity than single-source
algorithms.  Finally, there is the
<em>all-pairs shortest-paths problem</em>.
For this we want to find the shortest paths between all pairs
of vertices.  For dense graphs, one typically solves this problem with
the Floyd-Warshall algorithm
cite{cormen:2001}.
For sparse graphs,
Johnson's algorithm
cite{cormen:2001},
which computes the single-source
problem for each vertex, is asymptotically faster.




If the graph contains negative weight edges, then the shortest path between
two connected vertices may not be well defined.  This occurs if there is
a negative weight cycle reachable between the source and the destination.
Then one can construct a path between the source and the destination with
arbitrarily low weight by repeating the negative weight cycle.  Some
shortest-path algorithms, like the Bellman-Ford algorithm, are able to
detect negative weight cycles and then indicate that the
shortest-paths problem does not have a solution.
Other algorithms, like Dijkstra's algorithm, assume that the edge weights
are nonnegative.  We will consider only graphs with nonnegative weights.


We can represent the shortest-path weights from a given source by
storing this value as an attribute in each vertex.  The distance of the
source is defined to be zero (source.distance = 0), the
distance of unreachable vertices is infinite.  Some algorithms keep
track of the <em>status</em> of the vertices.  For the algorithms we
will consider, a vertex is in one of three states: \c KNOWN if
the distance is known to have the correct value, \c LABELED if
the distance has been updated from a known vertex and
\c UNLABELED otherwise.

The shortest paths form a tree with root at the source.  We can
represent this tree by having each vertex store a <em>predecessor</em>
attribute.  The predecessor of a vertex is that vertex which comes
directly before it in the shortest path from the source.  The
predecessor of the source and of unreachable vertices is defined to be
the special value \c NONE .  Note that while the shortest
distance is uniquely defined, the predecessor is not.  This is because
there may be multiple shortest paths from the source to a given
vertex.  Below is the procedure for initializing a graph to solve the
single-source shortest-paths problem for a specified source vertex.


\verbatim
initialize(graph, source):
  for vertex in graph.vertices:
    vertex.distance = Infinity
    vertex.predecessor = NONE
    vertex.status = UNLABELED
  source.distance = 0
  source.status = KNOWN \endverbatim


The algorithms that we will consider generate the shortest paths through a
process of <em>labeling</em>.
At each stage, the distance attribute
of each vertex is an upper bound on the shortest-path weight.  If the
distance is not infinite, then it is the sum of the edge weights on
some path from the source.  This approximation of the shortest-path
weight is improved by <em>relaxing</em>
along an edge.  For a known
vertex, we see if the distance to its neighbors can be improved by
going through its adjacent edges.  For a \c knownVertex with
an adjacent \c edge leading to a \c vertex , if
<tt>knownVertex.distance + edge.weight < vertex.distance</tt> then
we improve the approximation of \c vertex.distance by setting it
to <tt>knownVertex.distance + edge.weight</tt>.  We also update the
predecessor attribute.  The algorithms proceed by labeling the
adjacent neighbors of known vertices and freezing the value of labeled
vertices when they are determined to be correct.  At termination, the
distance attributes are equal to the shortest-path weights.  Below is
the procedure for labeling a single vertex and the procedure for
labeling the neighbors of a known vertex.


\verbatim
label(vertex, knownVertex, edgeWeight):
  if vertex.status == UNLABELED:
    vertex.status = LABELED
  if knownVertex.distance + edgeWeight < vertex.distance:
    vertex.distance = knownVertex.distance + edgeWeight
    vertex.predecessor = knownVertex
  return \endverbatim


\verbatim
labelAdjacent(knownVertex):
  for each edge of knownVertex leading to vertex:
    if vertex.status != KNOWN:
      label(vertex, knownVertex, edge.weight)
  return \endverbatim




<!---------------------------------------------------------------------------->
\section shortest_paths_introduction_test Test Problems
<!--\label{shortest paths, introduction, test problems}-->

For the purpose of evaluating the performance of the shortest path
algorithms we introduce a few simple test problems for weighted,
directed graphs.  The first problem is the <em>grid graph</em>.  The
vertices are arranged in a 2-D rectangular array.  Each vertex has four
adjacent edges to its neighboring vertices.  Vertices along the
boundary are periodically connected.  Next we consider a
<em>complete graph</em> in which each vertex has adjacent edges to
every other vertex.  Finally, we introduce the <em>random graph</em>.
Each vertex has a specified number of adjacent and incident edges.
These edges are selected through random shuffles.
The figure below shows examples of the three test
problems.


\image html TestProblems.jpg "Examples of the three test problems:  A 3 by 3 grid graph, a complete graph with 5 vertices and a random graph with 5 vertices and 2 adjacent edges per vertex are shown."
\image latex TestProblems.pdf "Examples of the three test problems:  A 3 by 3 grid graph, a complete graph with 5 vertices and a random graph with 5 vertices and 2 adjacent edges per vertex are shown." width=\textwidth
<!--\label{figure test problems}-->


The edge weights are uniformly, randomly distributed on a given
interval.  We characterize the distributions by the ratio of the upper
to lower bound of the interval.  For example, edge weights on the
interval [1 / 2..1] have a ratio of \f$R = 2\f$ and edge weights on the
interval [0..1] have an infinite ratio, \f$R = \infty\f$.
*/







//=============================================================================
//=============================================================================
/*!
\page shortest_paths_dijkstra Dijkstra's Greedy Algorithm




Dijkstra's
algorithm cite{cormen:2001} cite{cherkassky/goldberg/radzik:1996}
solves the single-source shortest-paths problem for the case that the
edge weights are nonnegative.  It is a labeling algorithm.  Whenever
the distance of a vertex becomes known, this known vertex labels its
adjacent neighbors.  The algorithm begins by labeling the adjacent
neighbors of the source.  The vertices with \c LABELED status
are stored in the \c labeled set.  The algorithm iterates until
the \c labeled set is empty.  This occurs when all vertices
reachable from the source become \c KNOWN .  At each step of the
iteration, the labeled vertex with minimum distance is guaranteed to
have the correct distance and a correct predecessor.  The status of
this vertex is set to \c KNOWN , it is removed from the
\c labeled set and its adjacent neighbors are labeled.  Below is
Dijkstra's algorithm.  The \c extractMinimum() function
removes and returns the vertex with minimum distance from the
\c labeled set.  We postpone the discussion of why the
\c labelAdjacent() function takes the \c labeled set as
an argument.


\verbatim
Dijkstra(graph, source):
  initialize(graph, source)
  labeled.clear()
  labelAdjacent(labeled, source)
  while labeled is not empty:
    minimumVertex = extractMinimum(labeled)
    minimumVertex.status = KNOWN
    labelAdjacent(labeled, minimumVertex)
  return \endverbatim


Dijkstra's algorithm is a greedy algorithm because at each step, the
best alternative is chosen.  That is, the labeled vertex with the
smallest distance becomes known.  Now we show how this greedy strategy
produces a correct shortest-paths tree.  Suppose that some of the
vertices are known to have the correct distance and that all adjacent
neighbors of these known vertices have been labeled.  We assert that
the labeled vertex \c v with minimum distance has the correct
distance.  Suppose that this distance to \c v is computed
through the path \c source \f$\mapsto\f$ \c x \f$\to\f$
\c v.  (Here \f$\mapsto\f$ indicates a (possibly empty) path and
\f$\to\f$ indicates a single edge.)  Each of the vertices in the path
\c source \f$\mapsto\f$ \c x is known.  We assume that there
exists a shorter path, \c source \f$\mapsto\f$ \c y \f$\to\f$
\c u \f$\mapsto\f$ \c v, where \c y is known and
\c u is labeled, and obtain a contradiction.  First note that
all paths from \c source to \c v have this form.  At some
point the path progresses from a known vertex \c y to a labeled
vertex \c u.  Since \c u is labeled, \c u.distance
\f$\geq\f$ \c v.distance .  Since the edge weights are nonnegative,
\c source \f$\mapsto\f$ \c y \f$\to\f$ \c u \f$\mapsto\f$
\c v is not a shorter path.  We conclude that \c v has the
correct distance.

Dijkstra's algorithm produces a correct shortest-paths tree.  After
initialization, only the source vertex is known.  At each step of
the iteration, one labeled vertex becomes known.  The algorithm
proceeds until all vertices reachable from the source have the correct
distance.

The figure below shows an example of using
Dijkstra's algorithm to compute the shortest paths tree.  First we
show the graph, which is a \f$3 \times 3\f$ grid graph except that the boundary
vertices are not periodically connected.  In the initialization step,
the lower left vertex is set to be the source and its two neighbors
are labeled.  We show known vertices in black and labeled vertices in
red.  The current labeling edges are green.  Edges of the shortest-paths
tree are shown in black, while the predecessor edges for labeled
vertices are red.  After initialization, there is one known vertex
(namely the source) and two labeled vertices.  In the first step, the
minimum vertex has a distance of 2.  This vertex becomes known and the
edge from its predecessor is added to the shortest paths tree.  Then
this vertex labels its adjacent neighbors.  In the second step, the
three labeled vertices have the same distance.  One of them becomes
known and labels its neighbors.  Choosing the minimum labeled vertex
and labeling its neighbors continues until all the vertices are known.


\image html DijkstraFigure.jpg "Dijkstra's algorithm for a graph with 9 vertices."
\image latex DijkstraFigure.pdf "Dijkstra's algorithm for a graph with 9 vertices." width=\textwidth
<!--\label{figure dijkstra figure}-->


Now that we have demonstrated the correctness of Dijkstra's algorithm,
we determine the computational complexity.  Suppose that we store the
labeled vertices in an array or a list.  If there are \e N labeled
vertices, the computational complexity of adding a vertex or decreasing
the distance of a vertex is \f$\mathcal{O}(1)\f$.  To extract the minimum
vertex we examine each labeled vertex for a cost of \f$\mathcal{O}(N)\f$.
There are \f$V - 1\f$ calls to \c push() and
\c extractMinimum() .  At any point in the shortest-paths
computation, there are at most \e V labeled vertices.  Hence the
\c push() and \c extractMinimum() operations add a
computational cost of \f$\mathcal{O}( V^2 )\f$.  There are \e E
labeling operations and less than \e E calls to \c decrease()
which together add a cost of \f$\mathcal{O}( E )\f$.  Thus the
computational complexity of Dijkstra's algorithm using an array or
list to store the labeled vertices is
\f$\mathcal{O}(V^2 + E) = \mathcal{O}( V^2)\f$


Now we turn our attention to how we can store the labeled vertices so that
we can more efficiently extract one with the minimum distance.
The \c labeled set is a priority queue cite{cormen:2001}
that supports three operations:
- \c push() :
  Vertices are added to the set when they become labeled.
- \c extractMinimum() :
  The vertex with minimum distance can be removed.
- \c decrease() :
  The distance of a vertex in the labeled set may be decreased through
  labeling.
.
For many problems, a binary heap cite{cormen:2001} is an efficient
way to implement the priority queue.  Below are new functions for
labeling vertices which now take the labeled set as an argument and
use the \c push() and \c decrease() operations on it.


\verbatim
labelAdjacent(heap, knownVertex):
  for each edge of knownVertex leading to vertex:
    if vertex.status != KNOWN:
      label(heap, vertex, knownVertex, edge.weight)
  return \endverbatim


\verbatim
label(heap, vertex, knownVertex, edgeWeight):
  if vertex.status == UNLABELED:
    vertex.status = LABELED
    vertex.distance = knownVertex.distance + edgeWeight
    vertex.predecessor = knownVertex
    heap.push(vertex)
  else if knownVertex.distance + edgeWeight < vertex.distance:
    vertex.distance = knownVertex.distance + edgeWeight
    vertex.predecessor = knownVertex
    heap.decrease(vertex.heapPointer)
  return \endverbatim


If there are \e N labeled vertices in the binary heap, the computational
complexity of adding a vertex is \f$\mathcal{O}(1)\f$.  The cost of
extracting the minimum vertex or decreasing the distance of a vertex
is \f$\mathcal{O}(\log N)\f$.  The \f$V - 1\f$ calls to \c push() and
\c extractMinimum() add a computational cost of
\f$\mathcal{O}( V \log V )\f$.  There are \e E labeling operations,
\f$\mathcal{O}( E )\f$, and less than \e E calls to \c decrease(),
\f$\mathcal{O}( E \log V )\f$.  Thus the computational complexity of
Dijkstra's algorithm using a binary heap is
\f$\mathcal{O}( ( V + E ) \log V )\f$.
*/





//=============================================================================
//=============================================================================
/*!
\page shortest_paths_greedier A Greedier Algorithm: Marching with a Correctness Criterion
<!--\label{chapter sssp section aga:mwacc}-->

If one were to solve by hand the single-source shortest-paths problem
using Dijkstra's algorithm, one would probably note that at any given
step, most of the labeled vertices have the correct distance and
predecessor.  Yet at each step only one vertex is moved from the labeled
set to the known set.  Let us quantify this observation.
At each step of Dijkstra's
algorithm (there are always \f$V - 1\f$ steps) we count the number of
correct vertices and the total number of vertices in the labeled set.
At termination we compute the fraction of vertices that had correct
values.  The fraction of correct vertices in the labeled set depends
on the connectivity of the vertices and the distribution of edge
weights.  As introduced in the
\ref shortest_paths_introduction_test "test problem section",
we consider grid, random
and complete graphs.  We consider edges whose weights have a uniform
distribution in a given interval.  The interval is characterized by
the ratio of its upper limit to its lower limit.  We consider the ratios:
2, 10, 100 and \f$\infty\f$.
The fractions of correctly determined labeled vertices are plotted below.
(The graphs show log-linear plots of the ratio of correctly determined
vertices to labeled vertices versus the number of vertices in the
graph.) We see that this fraction depends on the edge weight ratio.
This is intuitive.  If the edge weights were all unity (or another constant)
then we could solve the shortest-paths problem with a breadth first search.
At each iteration of Dijkstra's algorithm, all the labeled vertices would
have the correct value.  We see that as the edge weight ratio increases,
fewer of the labeled
vertices are correct, but even when the ratio is infinite a significant
fraction of the labeled vertices are correct.

\image html DijkstraDeterminedGrid.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex DijkstraDeterminedGrid.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html DijkstraDeterminedComplete.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex DijkstraDeterminedComplete.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html DijkstraDeterminedRandom4.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex DijkstraDeterminedRandom4.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html DijkstraDeterminedRandom32.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex DijkstraDeterminedRandom32.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth
<!--\label{figure dijkstra determined}-->


These observations motivate us to seek a new algorithm for the single-source
shortest-paths problem.  Dijkstra's algorithm is an example of a greedy
algorithm.  At each iteration the single best choice is taken.  The labeled
vertex with minimum distance is added to the known set.  We seek a greedier
algorithm.  At each iteration we take as many correct choices as possible.
Each labeled vertex that can be determined to have the correct distance is
added to the known set.  Below is this greedier algorithm.


\verbatim
marchingWithCorrectnessCriterion(graph, source):
  graph.initialize(source)
  labeled.clear()
  newLabeled.clear()
  labelAdjacent(labeled, source)
  // Loop until all vertices have a known distance.
  while labeled is not empty:
    for vertex in labeled:
      if vertex.distance is determined to be correct
        vertex.status = KNOWN
        labelAdjacent(newLabeled, vertex)
    // Get the labeled lists ready for the next step.
    removeKnown(labeled)
    labeled += newLabeled
    newLabeled.clear()
  return \endverbatim



It is easy to verify that the algorithm is correct.  It gives the correct
result because only vertices with correct distances are added to the
known set.  It terminates because at each iteration at least one vertex
in the labeled set has the correct distance.  We call this algorithm
<em>marching with a correctness criterion</em> (MCC).
All we lack now is a good method for determining if a labeled vertex
is correct.  The rest of the algorithm is trivial.  We do have one
correctness criterion, namely that used in Dijkstra's algorithm:  The labeled
vertex with minimum distance is correct.  Using this criterion would give
us Dijkstra's algorithm with a list as a priority queue, which has
computational complexity \f$\mathcal{O}( V^2 )\f$.
We turn our attention to finding a better correctness criterion.


Assume that some of the vertices are known to have the correct
distance and that all adjacent neighbors of known vertices have been
labeled.  To determine if a labeled vertex is correct, we look at the
labeling operations that have not yet occurred.  If future labeling
operations will not decrease the distance, then the distance must be
correct.  We formulate this notion by defining a lower bound on the
distance of a labeled vertex.  The distance stored in a labeled vertex
is an upper bound on the actual distance.  We seek to define a lower
bound on the distance by using the current distance and considering
future labeling operations.  If the current distance is less than or
equal to the lower bound, then the labeled vertex must be correct.  We
will start with a simple lower bound and then develop more
sophisticated ones.

Let \c minimumUnknown be the minimum distance among the labeled
vertices.  By the correctness criterion of Dijkstra's algorithm, any
labeled vertex with distance equal to \c minimumUnknown is
correct.  The simplest lower bound for a labeled vertex is the value
of \c minimumUnknown.  We call this the level 0 lower bound.

To get a more accurate lower bound, we use information about the
incident edges.  Let each vertex have the attribute
\c minimumIncidentEdgeWeight, the minimum weight over all
incident edges.  The smaller of \c vertex.distance and
<tt>(minimumUnknown + vertex.minimumIncidentEdgeWeight)</tt> is a
lower bound on the distance.  We consider why this is so.  If the
predecessor of this vertex in the shortest-paths tree is known, then
it has been labeled from its correct predecessor and has the correct
distance.  Otherwise, the distance at its correct predecessor is
currently not known, but is no less than \c minimumUnknown.  The
edge weight from the predecessor is no less than
\c vertex.minimumIncidentEdgeWeight.  Thus
<tt>(minimumUnknown + vertex.minimumIncidentEdgeWeight)</tt> is no
greater than the correct distance.  We call the minimum of
\c vertex.distance and <tt>(minimumUnknown + vertex.minimumIncidentEdgeWeight)</tt>
the level 1 lower bound.


\verbatim
lowerBound1(vertex, minimumUnknown)
  return min(vertex.distance, minimumUnknown + vertex.minimumIncidentEdgeWeight) \endverbatim


If the distance at a labeled vertex is less than or equal to the lower
bound on the distance, then the vertex must have the correct distance.
This observation allows us to define the level 1 correctness
criterion.  We define the \c isCorrect1() method for a
vertex.  For a labeled vertex, it returns true if the current distance
is less than or equal to the level 1 lower bound on the distance and
false otherwise.


\verbatim
isCorrect1(vertex, minimumUnknown, level)
  return (vertex.distance <= vertex.lowerBound1(minimumUnknown)) \endverbatim


Below we show an example of using the MCC
algorithm with the level 1 correctness criterion to compute the
shortest-paths tree.  First we show the graph, which is a \f$4 \times 4\f$
grid graph except that the boundary vertices are not periodically
connected.  In the initialization step, the lower, left vertex is set
to be the source and its two neighbors are labeled.  We show known
vertices in black and labeled vertices in red.  The labeling
operations are shown in green.  Edges of the shortest paths tree are
shown in black, while the predecessor edges for labeled vertices are
red.  After initialization, there is one known vertex (namely the
source) and two labeled vertices.  Depictions of applying the
correctness criterion are shown in blue.  (Recall that the level 1
correctness criterion uses the minimum incident edge weight and the
minimum labeled vertex distance to determine if a labeled vertex is
correct.)  Since future labeling operations will not decrease their
distance, both labeled vertices become known in the first step.  After
labeling their neighbors, there are three labeled vertices in step 1.
The correctness criterion shows that the vertices with distances 3 and
4 will not be decreased by future labeling operations, thus they are
correct.  However, the correctness criterion does not indicate that
the labeled vertex with distance 8 is correct.  The correctness
criterion indicates that a vertex with a distance as small as 3 might
label the vertex with an edge weight as small as 2.  This gives a
lower bound on the distance of 5.  Thus in step 2, two of the three
labeled vertices become known.  We continue checking labeled vertices
using the correctness criterion until all the vertices are known.
Finally, we show the shortest-paths tree.


\image html MCCFigure.jpg "Marching with a correctness criterion algorithm for a graph with 16 vertices."
\image latex MCCFigure.pdf "Marching with a correctness criterion algorithm for a graph with 16 vertices." width=\textwidth
<!--\label{figure mcc figure}-->


We can get a more accurate lower bound on the distance of a labeled vertex
if we use more information about the incident edges.  For the level 1 formula,
we used only the minimum incident edge weight.  For the level 2 formula
below we use all of the unknown incident edges.
\f[
\min \left( \mathtt{vertex.distance},
  \min_{\substack{\mathrm{unknown}\\ \mathrm{edges}}} (\mathtt{edge.weight + minimumUnknown})
\right)
\f]
The lower bound is the smaller of the current distance and the minimum
over unknown incident edges of <tt>(edge.weight + minimumUnknown)</tt>.
Let the method <tt>lowerBound(minimumUnknown, level)</tt> return the lower
bound for a vertex.  Since the level 0 lower bound is \c minimumUnknown,
we can write the level 2 formula in terms of the level 0 formula.
\f[
  \mathtt{vertex.lowerBound( minimumUnknown, 2 )} =
\f]
\f[
  \min \left( \mathtt{vertex.distance},
  \min_{\substack{\mathrm{unknown}\\ \mathrm{edges}}} (\mathtt{edge.weight
    + edge.source.lowerBound( minimumUnknown, 0 )}) \right)
\f]
More generally, for \f$n \geq 2\f$ we can define the level \e n lower bound in terms
of the level \e n - 2 lower bound.  This gives us a recursive definition of
the method.
\f[
  \mathtt{vertex.lowerBound( minimumUnknown, n )} =
\f]
\f[
  \min \left( \mathtt{vertex.distance},
    \min_{\substack{\mathrm{unknown}\\ \mathrm{edges}}} (\mathtt{edge.weight
      + edge.source.lowerBound( minimumUnknown, n - 2 )}) \right)
\f]
We consider why this is a correct lower bound.  If the correct
predecessor of this vertex is known, then it has been labeled from its
predecessor and thus has the correct distance.  Otherwise, the
distance at its predecessor is currently not known.  The correct
distance is the correct distance of the predecessor plus the weight of
the connecting, incident edge. The minimum over unknown edges of the
sum of edge weight and a lower bound on the distance of the incident
vertex is no greater than the correct distance.  Thus the lower bound
formula is valid.  Below is the \c lowerBound method which
implements the lower bound formulae.


\verbatim
lowerBound(vertex, minimumUnknown, level)
  if level == 0:
    return minimumUnknown
  if level == 1:
    return min(vertex.distance, minimumUnknown + vertex.minimumIncidentEdgeWeight)
  minimumDistance = vertex.distance
  for edge in vertex.incidentEdges:
    if edge.source.status != KNOWN:
      d = edge.weight + edge.source.lowerBound(minimumUnknown, level - 2)
      if d < minimumDistance:
        minimumDistance = d
  return minimumDistance \endverbatim


Now we define the \c isCorrect() method for a vertex.  For a labeled
vertex, it returns true if the current distance is less than or equal to
the lower bound on the distance and false otherwise.  This completes the
\c marchingWithCorrectnessCriterion() function.  We give the refined
version of this function below.


\verbatim
isCorrect(vertex, minimumUnknown, level)
  return (vertex.distance <= vertex.lowerBound(minimumUnknown, level)) \endverbatim


\verbatim
marchingWithCorrectnessCriterion(graph, source, level)
  graph.initialize(source)
  labeled.clear()
  newLabeled.clear()
  labelAdjacent(labeled, source)
  // Loop until all vertices have a known distance.
  while labeled is not empty:
    minimumUnknown = minimum distance in labeled
    for vertex in labeled:
      if vertex.isCorrect(minimumUnknown, level):
        vertex.status = KNOWN
        labelAdjacent(newLabeled, vertex)
    // Get the labeled lists ready for the next step.
    removeKnown(labeled)
    labeled += newLabeled
    newLabeled.clear()
  return \endverbatim


The figure below depicts the incident edges used for the
first few levels of correctness criteria.  For each level, the
correctness criterion is applied to the center vertex.  We show the
surrounding vertices and incident edges.  The level 0 criterion does
not use any information about the incident edges.  The level 1
criterion uses only the minimum incident edge.  The level 2 criterion
uses all the incident edges from unknown vertices.  The level 3
criterion uses the incident edges from unknown vertices and the
minimum incident edge at each of these unknown vertices.  The figure
depicts subsequent levels up to level 6.  If each vertex had \e I
incident edges then the computational complexity of the level \e n
correctness criterion would be \f$\mathcal{O}(I^{\lfloor n/2 \rfloor})\f$.


\image html MCCLevel.jpg "A depiction of the incident edges used in the level n correctness criterion for n = 0, ..., 6."
\image latex MCCLevel.pdf "A depiction of the incident edges used in the level n correctness criterion for n = 0, ..., 6." width=\textwidth
<!--\label{figure mcc level}-->


We examine the performance of these correctness criteria.  From our analysis
of correctly determined vertices in Dijkstra's algorithm
we expect that the ratio of vertices
which can be determined to be correct will depend on the connectivity of
the edges and the distribution of edge weights.  We also expect that for a
given graph, the ratio of vertices which are determined to be correct will
increase with the level of the correctness criterion.
Again we consider grid, random
and complete graphs with maximum-to-minimum edge weight ratios of
2, 10, 100 and \f$\infty\f$.




The graphs below show the performance of the
correctness criteria for each kind of graph with an edge weight ratio
of 2.  We run the tests for level 0 through level 5 criteria.  We
show the ideal algorithm for comparison.  (The ideal correctness
criterion would return true for all labeled vertices whose current
distance is correct.)  The level 0 correctness criterion
(which is the criterion used in Dijkstra's algorithm) yields a very
low ratio of correctly determined vertices.  If the minimum weight of
labeled vertices is unique, then only a single labeled vertex will
become determined.  The level 1 and level 2 criteria perform quite
well.  For the grid graphs and random graphs, about 3/4 of the
labeled vertices are determined at each step.  For the complete graph,
all of the labeled vertices are determined at each step.  For the
ideal criterion, the ratio of determined vertices is close to or equal
to 1 and does not depend on the number of vertices.  The correctness
criteria with level 3 and higher come very close to the ideal
criterion.  We see that the level 1 and level 3 criteria are the most
promising for graphs with low edge weight ratios.  The level 1
criterion yields a high ratio of determined vertices.  The level 2
criterion yields only marginally better results at the cost of greater
algorithmic complexity and higher storage requirements.  Recall that
the level 1 criterion only uses the minimum incident edge weight,
while the level 2 criterion requires storing the incident edges at
each vertex.  The level 3 criterion comes very close to the ideal.
Higher levels only add complexity to the algorithm.


\image html MCCDeterminedGrid2.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex MCCDeterminedGrid2.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html MCCDeterminedComplete2.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex MCCDeterminedComplete2.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html MCCDeterminedRandom4Edges2.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex MCCDeterminedRandom4Edges2.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html MCCDeterminedRandom32Edges2.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex MCCDeterminedRandom32Edges2.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Next we show the performance of the correctness
criteria for each kind of graph with an edge weight ratio of 10.  Compared
to the results for an edge weight ratio of 2, the determined ratio is
lower for the ideal criterion and the determined ratios for levels 0 through 5
are more spread out.  Around 3/4 of the vertices are determined at each time
step with the ideal criterion; there is a slight dependence on the number
of vertices.  The level 1 criterion performs fairly well;
it determines from about 1/4 to 1/2 of the vertices.  The level 2
criterion determines only slightly more vertices than the level 1.  Going to
level 3 takes a significant step toward the ideal.   Level 4 determines few
more vertices than level 3.  There is a diminishing return in going to higher
levels.


\image html MCCDeterminedGrid10.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex MCCDeterminedGrid10.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html MCCDeterminedComplete10.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex MCCDeterminedComplete10.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html MCCDeterminedRandom4Edges10.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex MCCDeterminedRandom4Edges10.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html MCCDeterminedRandom32Edges10.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex MCCDeterminedRandom32Edges10.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


The graphs below show the performance of the
correctness criteria when the edge weight ratio is 100.  The
determined ratio is lower still for the ideal criterion and the
determined ratios for levels 0 through 5 are even more spread out.
The determined ratios now have a noticeable dependence on the number of
vertices.


\image html MCCDeterminedGrid100.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex MCCDeterminedGrid100.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html MCCDeterminedComplete100.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex MCCDeterminedComplete100.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html MCCDeterminedRandom4Edges100.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex MCCDeterminedRandom4Edges100.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html MCCDeterminedRandom32Edges100.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex MCCDeterminedRandom32Edges100.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Finally, we show the
performance of the correctness criteria with an infinite edge weight
ratio.  Compared to the results for lower edge weight ratios, the
determined ratio is lower for the ideal criterion and the determined
ratios for levels 0 through 5 are more spread out.  For the infinite
edge weight ratio, the correctness criteria yield fewer correctly
determined vertices.


\image html MCCDeterminedGridInfinity.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex MCCDeterminedGridInfinity.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html MCCDeterminedCompleteInfinity.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex MCCDeterminedCompleteInfinity.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html MCCDeterminedRandom4EdgesInfinity.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex MCCDeterminedRandom4EdgesInfinity.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html MCCDeterminedRandom32EdgesInfinity.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex MCCDeterminedRandom32EdgesInfinity.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Note that if the correctly determined ratio is \e D, then on average a
labeled vertex will be tested 1 / \e D times before it is determined to
be correct.  For each of the correctness criteria, we see that the
ratio of determined vertices is primarily a function of the edge
weight ratio and the number of edges per vertex.  The determined ratio
decreases with both increasing edge weight ratio and increasing edges
per vertex.  Thus graphs with a low edge weight ratio and/or few edges
per vertex seem well suited to the marching with a correctness
criterion approach.  For graphs with high edge weight ratios and/or
many edges per vertex, the correctness criteria yield fewer determined
vertices, so the method will be less efficient.


Before analyzing execution times in the next section, we develop a more
efficient implementation of the correctness criteria for levels 2 and higher.
It is not necessary to examine all of the edges of a vertex each time
\c isCorrect() is called.  Instead, we amortize this cost
over all the calls.  The incident edges of each vertex are in sorted order
by edge weight.  Additionally, each vertex has a forward edge iterator,
\c unknownIncidentEdge,
which keeps track of the incident edge currently being considered.
To see if a labeled vertex is determined,
the incident edges are traversed in order.  If we encounter an edge from
an unknown vertex such that the sum of the edge weight and the lower bound
on the distance of that unknown vertex is less than the current distance,
then the vertex is not determined to be correct.  The next time
\c isCorrect() is called for the vertex, we start at the incident
edge where the previous call stopped.  This approach works because
the lower bound on the distance of each vertex is non-increasing as the
algorithm progresses.  That is, as more vertices become known and more
vertices are labeled, the lower bound on a given vertex may decrease but
will never increase.    Below is the more efficient implementation of
\c isCorrect().


\verbatim
isCorrect(vertex, minimumUnknown, level):
  if level <= 1:
    if vertex.distance > vertex.lowerBound(minimumUnknown, level):
      return false
  else:
    vertex.getUnknownIncidentEdge()
    while vertex.unknownIncidentEdge != vertex.incidentEdges.end():
      if (vertex.distance > vertex.unknownIncidentEdge.weight + vertex.unknownIncidentEdge.source.lowerBound(minimumUnknown, level - 2)):
        return false
      ++vertex.unknownIncidentEdge
      vertex.getUnknownIncidentEdge()
  return true \endverbatim


\verbatim
getUnknownIncidentEdge(vertex):
  while (vertex.unknownIncidentEdge != vertex.incidentEdges.end() and vertex.unknownIncidentEdge.source.status == KNOWN):
    ++vertex.unknownIncidentEdge
  return \endverbatim
*/







//=============================================================================
//=============================================================================
/*!
\page shortest_paths_complexity Computational Complexity

Now we determine the computational complexity of the MCC algorithm.
We will get a worst-case bound for using the level 1 correctness
criterion.  Let the edge weights be in the interval \f$[A \ldots B]\f$.
As introduced before, let \f$R = B / A\f$ be the ratio of the largest
edge weight to the smallest.  We will assume that the ratio is finite.
Consider the MCC algorithm in progress.  Let \f$\mu\f$ be the minimum distance
of the labeled vertices.  The distances of the labeled vertices are in the
range \f$[\mu \ldots \mu + B)\f$.  When one applies the correctness criterion, at least
all of the labeled vertices with distances less than or equal to
\f$\mu + A\f$ will become known.  Thus at the next step, the minimum labeled
distance will be at least \f$\mu + A\f$.  At each step of the algorithm,
the minimum labeled distance increases by at least \e A.  This means
that a vertex may be in the labeled set for at most \f$B / A\f$ steps.  The
cost of applying the correctness criteria is
\f$\mathcal{O}(R V)\f$.  The cost of labeling is \f$\mathcal{O}(E)\f$.  Since
a vertex is simply added to the end of a list or array when it becomes
labeled, the cost of adding and removing labeled vertices is
\f$\mathcal{O}(V)\f$.  Thus the computation complexity of the MCC algorithm
is \f$\mathcal{O}(E + R V)\f$.
*/



//=============================================================================
//=============================================================================
/*!
\page shortest_paths_performance Performance Comparison



We compare the performance of Dijkstra's algorithm and the Marching
with a Correctness Criterion algorithm.  For the MCC algorithm, we
consider level 1 and level 3 correctness criteria, which have better
performance than other levels.  Again we consider grid, random and
complete graphs with maximum-to-minimum edge weight ratios of 2,
10, 100 and \f$\infty\f$.  The graphs below show the
execution times over a range of graph sizes for each kind of graph
with an edge weight ratio of 2.  The level 1 MCC algorithm has
relatively low execution times.  It performs best for sparse
graphs. (The grid graph and the first random graph each have four
adjacent and four incident edges per vertex.)  For the random graph
with 32 edges, it is still the fastest method, but the margin is
smaller.  For medium to large complete graphs, the execution times are
nearly the same as for Dijkstra's algorithm.  For small complete
graphs, Dijkstra's algorithm is faster.  The level 3 MCC algorithm
performs pretty well for the sparser graphs, but is slower than the
other two methods for the denser graphs.



\image html ExecutionTimeGrid2.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex ExecutionTimeGrid2.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html ExecutionTimeComplete2.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex ExecutionTimeComplete2.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html ExecutionTimeRandom4Edges2.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex ExecutionTimeRandom4Edges2.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html ExecutionTimeRandom32Edges2.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex ExecutionTimeRandom32Edges2.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Next we show the
execution times for graphs with an edge weight ratio of 10.
Again the level 1 MCC algorithm has the best overall performance, however
the margin is a little smaller than in the previous tests.


\image html ExecutionTimeGrid10.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex ExecutionTimeGrid10.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html ExecutionTimeComplete10.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex ExecutionTimeComplete10.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html ExecutionTimeRandom4Edges10.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex ExecutionTimeRandom4Edges10.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html ExecutionTimeRandom32Edges10.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex ExecutionTimeRandom32Edges10.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


The graphs below show the
execution times for graphs with an edge weight ratio of 100.
The level 1 MCC algorithm is no longer the best overall performer.
It has about the same execution times as Dijkstra's algorithm for the
complete graph and the random graph with 32 edges, but is slower than
Dijkstra's algorithm for the grid graph and the random graph with 4 edges.


\image html ExecutionTimeGrid100.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex ExecutionTimeGrid100.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html ExecutionTimeComplete100.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex ExecutionTimeComplete100.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html ExecutionTimeRandom4Edges100.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex ExecutionTimeRandom4Edges100.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html ExecutionTimeRandom32Edges100.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex ExecutionTimeRandom32Edges100.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Finally we show the execution times
for graphs with an infinite edge weight ratio.  Except for complete
graphs, the level 1 MCC algorithm does not scale well as the number of
vertices is increased.  This makes sense upon examining the correctly
determined ratio plots for an infinite edge weight.
<!--CONTINUE in Figure ref{figure mcc determined infinity}-->
As the size of the graph increases, the determined ratio decreases.
The level 3 MCC algorithm scales better, but is slower than Dijkstra's
algorithm for each size and kind of graph.


\image html ExecutionTimeGridInfinity.jpg "Grid graph. Each vertex has an edge to its four adjacent neighbors."
\image latex ExecutionTimeGridInfinity.pdf "Grid graph. Each vertex has an edge to its four adjacent neighbors." width=0.5\textwidth

\image html ExecutionTimeCompleteInfinity.jpg "Dense graph. Each vertex has an edge to every other vertex."
\image latex ExecutionTimeCompleteInfinity.pdf "Dense graph. Each vertex has an edge to every other vertex." width=0.5\textwidth

\image html ExecutionTimeRandom4EdgesInfinity.jpg "Random graph. Each vertex has edges to 4 randomly chosen vertices."
\image latex ExecutionTimeRandom4EdgesInfinity.pdf "Random graph. Each vertex has edges to 4 randomly chosen vertices." width=0.5\textwidth

\image html ExecutionTimeRandom32EdgesInfinity.jpg "Random graph. Each vertex has edges to 32 randomly chosen vertices."
\image latex ExecutionTimeRandom32EdgesInfinity.pdf "Random graph. Each vertex has edges to 32 randomly chosen vertices." width=0.5\textwidth


Note that the four plots for complete graphs (with edge weight ratios
of 2, 10, 100 and \f$\infty\f$) are virtually identical.
Dijkstra's algorithm and the level 1 MCC algorithm both perform
relatively well.  For medium to large graphs, their execution times are
very close.  This is because there are many more edges than vertices,
so labeling adjacent vertices dominates the
computation.  The costs of heap operations for Dijkstra's algorithm or
correctness tests for the level 1 MCC algorithm are negligible.  This
is the case even for an infinite edge weight ratio where on average,
each labeled vertex is checked many times before it is determined to
be correct.  For complete graphs, the level 3 MCC algorithm is slower
than the other two.  This is because all incident edges of a labeled
vertex may be examined during a correctness check.  Thus the level 3
correctness criterion is expensive enough to affect the execution
time.


Clearly the distribution of edge weights affects the performance of
the MCC algorithm.  One might try to use topological information to
predict the performance.  Consider planar graphs for example,
i.e. graphs that can be drawn in a plane without intersecting edges.
During the execution of the MCC algorithm (or Dijkstra's algorithm)
one would expect the labeled vertices to roughly form a band that
moves outward from the source.  In this case, the number of labeled
vertices would be much smaller than the total number of vertices.  One
might expect that the MCC algorithm would be well suited to planar
graphs.  However, this is not necessarily the case.
The figure below shows two shortest-paths
trees for a grid graph (a graph in which each vertex is connected to
its four adjacent neighbors).  The first diagram shows a typical tree.
The second diagram shows a pathological case in which the
shortest-paths tree is a single path that winds through the graph.
For this case, the average number of labeled vertices is of the order
of the number of vertices.  Also, at each step of the MCC algorithm
only a single labeled vertex can become known.  For this pathological
case, the complexity of the MCC algorithm is \f$\mathcal{O}(N^2)\f$.
Unfortunately, one cannot guarantee reasonable performance of the MCC
algorithm based on topological information.

\image html GridGraphPathological.jpg "Two shortest-paths trees for a grid graph."
\image latex GridGraphPathological.pdf "Two shortest-paths trees for a grid graph." width=\textwidth
*/







//=============================================================================
//=============================================================================
/*!
\page shortest_paths_concurrency Concurrency

Because of its simple data structures, the MCC method is easily
adapted to a concurrent algorithm.  The outer loop of the algorithm,
<tt>while labeled is not empty</tt>, contains two loops over the
labeled vertices.  The first loop computes the minimum labeled
distance and assigns this value to \c minimumUnknown.  Though the
time required to determine this minimum unknown distance is small
compared to correctness checking and labeling, it may be done
concurrently.  The complexity of finding the minimum of \e N elements
with \e P processors is \f$\mathcal{O}(N/P + \log_2 P)\f$
cite{vandevelde:1994}.
The more costly operations are contained in the second loop.  Each
labeled vertex is tested to see if its distance can be determined to
be correct.  If so, it becomes known and labels its adjacent
neighbors.  These correctness checks and labeling may be done
concurrently.  Thus the complexity for both is then
\f$\mathcal{O}(N/P)\f$.  We conclude that the computational complexity of the
MCC algorithm scales well with the number of processors.

By contrast, most other shortest-paths algorithms, including
Dijkstra's algorithm, are not so easily adapted to a concurrent
framework.  Consider Dijkstra's algorithm: The only operation that
easily lends itself to concurrency is labeling vertices when a labeled
vertex becomes known.  That is, labeling each adjacent vertex is an
independent operation.  These may be done concurrently.  However, this
fine scale concurrency is limited by the number of edges per vertex.
Because only one vertex may become known at a time, Dijkstra's
algorithm is ill suited to take advantage of concurrency.
*/



//=============================================================================
//=============================================================================
/*!
\page shortest_paths_future Future Work
<!--\label{chapter sssp section fw}-->


<!-------------------------------------------------------------------------->
\section shortest_paths_future_data A More Sophisticated Data Structure for the Labeled Set

There are many ways that one could adapt the MCC algorithm.  The MCC
algorithm has a sophisticated correctness criterion and stores the
labeled set in a simple container (namely, an array or a list).  At
each step the correctness criterion is applied to all the labeled
vertices.  By contrast, Dijkstra's algorithm has a simple correctness
criterion and stores the labeled set in a sophisticated container.  At
each step the correctness criterion is applied to a single
labeled vertex.  An approach that lies somewhere between these two
extremes may work well for some problems.  That approach would be to
employ both a sophisticated correctness criterion and a sophisticated
vertex container.

This more sophisticated container would need to be able to efficiently
identify the labeled vertices on which the correctness test is likely
to succeed.  At each step, the correctness criterion would be applied
to this subset of the labeled vertices.  For example, the labeled set
could be stored in a cell array, cell sorted by distance.  The
correctness criterion would be applied to vertices whose current
distance is less than a certain threshold, as vertices with small
distances are more likely to be correct than vertices with large
distances.  Alternatively, the labeled vertices could be cell sorted
by some other quantity, perhaps the difference of the current distance
and the minimum incident edge weight from an unknown vertex.  Again,
vertices with lower values are more likely to be correct.  Yet another
possibility would be to factor in whether the distance at a vertex
decreased during the previous step.  In summary, there are many
possibilities for partially ordering the labeled vertices to select a
subset to be tested for correctness.  A more sophisticated data
structure for storing the labeled set may improve performance,
particularly for harder problems.


One can use a cell array data structure to reduce the computational
complexity of the MCC algorithm for the case that the edge weight
ratio \e R is finite.  As introduced before, let the edge weights be in
the interval \f$[A .. B]\f$.  Each cell in the cell array holds the
labeled vertices with distances in the interval \f$[ n A .. (n+1) A)\f$
for some integer \e n.  Consider the MCC algorithm in progress.  Let
\f$\mu\f$ be the minimum labeled distance.  The labeled distances are in
the range \f$[\mu .. \mu + B)\f$.  We define \f$m = \lfloor \mu / A
\rfloor\f$. The first cell in the cell array holds labeled vertices in
the interval \f$[ m A .. (m+1) A)\f$.  By the level 1 correctness
criterion, all the labeled vertices in this cell are correct.  We
intend to apply the correctness criterion only to the labeled vertices
in the first cell.  If they labeled their neighbors, the neighbors
would have distances in the interval \f$[\mu + A .. \mu + A + B)\f$.  Thus
we need a cell array with \f$\lceil R \rceil + 1\f$ cells in order to span
the interval \f$[\mu .. \mu + A + B)\f$.  (This interval contains all the
currently labeled distances and the labeled distances resulting from
labeling neighbors of vertices in the first cell.)  At each step of
the algorithm, the vertices in the first cell become known and label
their neighbors.  If an unlabeled vertex becomes labeled, it is added
to the appropriate cell.  If a labeled vertex decreases its distance,
it is moved to a lower cell.  After the labeling, the first cell is
removed and an empty cell is added at the end.  As Dijkstra's
algorithm requires that each labeled vertex stores a pointer into the
heap of labeled vertices, this modification of the MCC algorithm would
require storing a pointer into the cell array.

Now consider the computational complexity of the MCC algorithm that uses
a cell array to store the labeled vertices.  The complexity of adding
or removing a vertex from the labeled set is unchanged, because the
complexity of adding to or removing from the cell array is \f$\mathcal{O}(1)\f$.
The cost of decreasing the distance of a labeled vertex is unchanged
because moving a vertex in the cell array has cost \f$\mathcal{O}(1)\f$.
We reduce the cost of applying the correctness criterion from
\f$\mathcal{O}(R V)\f$ to \f$\mathcal{O}(V)\f$ because each vertex is ``tested''
only once.  We must add the cost of examining cells in the cell array.
Let \e D be the maximum distance in the shortest path tree.
Then in the course of the computation, \f$D/A\f$ cells will be examined.
The total computational complexity of the MCC algorithm with a cell array
for the labeled vertices is \f$\mathcal{O}(E + V + D/A)\f$.  Note that
\e D could be as large as \f$(V-2)B + A\f$.  In this case \f$D/A \approx R V\f$ and the
computational complexity is the same as that for the plain MCC algorithm.





<!-------------------------------------------------------------------------->
\section shortest_paths_future_reweighting Re-weighting the Edges

Let <em>w(u,v)</em> be the weight of the edge from vertex \e u to vertex \e v.
Consider a function \f$f : V \to \mathbb{R}\f$ defined on the vertices and
a modified weight function \f$\hat{w}\f$:
\f[
\hat{w}(u,v) = w(u,v) + f(u) - f(v)
\f]
It is straightforward to show that any shortest path with weight
function \e w is also
a shortest path with weight function \f$\hat{w}\f$ cite{cormen:2001}.
%% Section 25.3
This is because \e any path from vertex \e a to vertex \e b is changed by
\f$f(a) - f(b)\f$.  The rest of the \e f terms telescope in the sum.

It may be possible to re-weight the edges of a graph to improve the
performance of the MCC algorithm.  The number of correctness tests performed
depends on the ratio of the highest to lowest edge weight.  By choosing \e f
to decrease this ratio, one could decrease the execution time.
One might determine the function \e f as a preprocessing step or perhaps
compute it on the fly as the shortest-paths computation progresses.

Consider the situation at a single vertex.  Assume that \e f is zero at all
other vertices.  Let \c minimumIncident, \c minimumAdjacent,
\c maximumIncident and \c maximumAdjacent denote the minimum and
maximum incident and adjacent edge weights.  If
<tt>minimumAdjacent < minimumIncident</tt> and
<tt>maximumAdjacent < maximumIncident</tt>, then choosing
\f[
f = \frac{\mathtt{minimumIncident} - \mathtt{minimumAdjacent}}{2}
\f]
will reduce the ratio of the highest to lowest edge weight at the given
vertex.  Likewise for the case:
<tt>minimumAdjacent > minimumIncident</tt> and
<tt>maximumAdjacent > maximumIncident</tt>.
*/



//=============================================================================
//=============================================================================
/*!
\page shortest_paths_conclusions Conclusions



Marching with a Correctness Criterion is a promising new approach for
solving shortest path problems.  MCC works well on easy problems.
That is, if most of the labeled vertices are correct, then the
algorithm is efficient.  It requires few correctness tests before a
vertex is determined to be correct.  For such cases, the MCC algorithm
outperforms Dijkstra's algorithm.  For hard problems, perhaps in
which the edge weight ratio is high and/or there are many edges per
vertex, fewer labeled vertices have the correct distance.  This means
that the MCC algorithm requires more correctness tests.  For such
cases, Dijkstra's algorithm has lower execution times.
*/



#endif

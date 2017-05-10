// -*- C++ -*-

#if !defined(__particle_h__)
#define __particle_h__

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"
#include "stlib/particle/adjacent.h"
#include "stlib/particle/verlet.h"

/*!
  \file particle.h
  \brief Computational geometry and concurrency for particle methods.
*/

namespace stlib
{
//! All classes and functions in the particle package are defined in the particle namespace.
namespace particle
{

//=============================================================================
//=============================================================================
/*!
\mainpage Particle Methods

<!------------------------------------------------------------------------>
\section particle_motivation Motivation

\par Unordered.
Consider a particle simulation where particles interact with their
neighbors (those within a specified distance). Below is a diagram for a 2D
example whose domain is an equilateral box. The bar at the bottom, which is
colored by hue, represents how the particles are laid out in memory. They
are stored in a contiguous array. Above that, we show the physical locations
for the particles. The grid lines are only shown for reference. Note that
the physical location of a particle is unrelated to its location in memory.

\image html particlesUnordered.png "Particles, unordered."
\image latex particlesUnordered.png "Particles, unordered."

\par Sorted.
Next we consider a simulation in which the particles are sorted by their
y-coordinates. The figure below illustrates this. Note that now there is
some correlation between the physical location and the memory location.
This ordering improves data locality for neighbor calculations for two
reasons: Firstly, the neighbors of a particle are, on average, closer in
memory than for an unordered sequence. Secondly, after performing a calculation
with one particle and its neighbors, the next particle to be used (and
its neighbors) are closer in memory. Of course, for the second point to
be valid, one must process the particles in order.

\image html particlesSorted.png "Particles, sorted."
\image latex particlesSorted.png "Particles, sorted."

\par
Above we sorted to achieve an ordering of the particles with improved
data locality. However, we only used one coordinate for the ordering.
Thus, there is a correlation between a particle's \e y coordinate and its
memory location, but none for the \e x coordinate. For example, the first
particle above is in the lower \e right corner, while the second particle
is in the lower \e left corner.

\par Morton order.
There are orderings that use all coordinates to improve data locality.
The domain is recursively divided into cells, and then the cells are ordered.
Tracing the order of the cells yields the associated
<i>space-filling curve</i>. Below we show a particle simulation with
a %Morton ordering. Now the grid lines define the cells, and we have traced
the %Morton curve to aid in visualizing the order.

\image html particlesMorton.png "Particles, Morton order."
\image latex particlesMorton.png "Particles, Morton order."

\par Squared distance.
For most problems, data layout is second only to asymptotic
computational complexity in importance when optimizing performance. As
an example, we compute the squared distance to each neighbor for a set of
particles in 3D.  That is, after determining the neighbors, we compute the
distance to each one. Of course, in a real simulation, one would use
the distance and direction to apply forces, etc. We consider
unordered, sorted order, and the %Morton order. Below we show the
average time to compute the squared distance per neighbor, in
nanoseconds. We vary the number of particles from one thousand to four
million. For each test, the interaction distance is chosen so that
each particle has about 30 neighbors.  The tests were run on a MacBook
Pro with a 2.8 GHz Intel Core 2 Duo processor, 8 GB of 1067 MHz DDR3
memory, and 6 MB L2 cache.

\image html neighborsSqDist.jpg "Time to compute squared distance."
\image latex neighborsSqDist.pdf "Time to compute squared distance."

\par
First note that for unordered particles, the calculation has very poor
scalability.  Everything is fine, until you hit 262,144 particles. At
that point, the particles and packed neighbor lists no longer fit in
the L2 cache.  As you increase the problem size, cache misses become
the dominant cost.  Of course, the tests with a large number of
particles are the interesting ones.  It is rare to run a practical
simulation in which all of the data fits in the L2 cache.  Sorting the
particles by the \e z coordinate dramatically improves the
scalability. The %Morton order, which has the best data locality,
offers the best performance. For this ordering, the time per neighbor is
nearly constant as one increases the problem size.

\par Summary.
We have seen that for particle simulations, like many other scientific computing
applications, the data layout is critically important for performance.
Using the %Morton order gave our calculation excellent scalability.
In the following, we will see that these concepts of ordering, specifically
spatial indexing, may be used for much more. We will introduce a tree
data structure that not only yields good data locality for particle
calculations, but may be used to determine the neighbors, and also may
be used to dynamically partition particles for distributed-memory
architectures.


<!------------------------------------------------------------------------>
\section particle_introduction Introduction

\par
This package is a framework for particle methods that
performs the following tasks:
- Determine the neighbors for each particle.
- Order the particles to improve data locality when performing calculations
with neighbors.
- Partition particles for distributed-memory architectures.
- Exchange particles at each time step for distributed-memory architectures.

\par
The framework takes ownership of the particles and stores them in a
\c std::vector. The user can access them through a public data member.
Between time steps, the particles may be reordered to maintain good
data locality and the data structure for calculating neighbors may be
rebuilt. In addition, for distributed memory applications, particles may
migrate between sockets, or be redistributed amongst the sockets.

\par
In the following, we will describe how particles are stored in a linear
orthtree using %Morton codes, and how we use this tree to perform various
calculations. However, to use this package you won't work directly
with the tree or the codes associated with particles. Sections that provide
these implementation details are marked with an asterisk. You may skim or
skip them if you like. As far as the user is concerned, the framework
provides a \c std::vector of particles and an interface for accessing
neighbor information. The user takes care of the physics, the framework
handles the computational geometry and concurrency.

\par
There are unit tests which exercise this code in \c stlib/test/unit/particle.
There are example programs that illustrate its use in
\c stlib/test/performance/particle.

\par
Below we show an example of how to use the framework.
We assume that the particle class, \c Particle, a functor that returns
a particles position, \c GetPostion, and the interaction distance,
\c interactionDistance, have been defined.

\par
\code
// 3-D space.
const std::size_t Dimension = 3;
// We will use single-precision floating point numbers.
typdef float Float;
// Define the traits for a plain (non-periodic) domain.
typedef particle::PlainTraits<Particle, GetPosition, Dimension, Float> Traits;
// The class that orders the particles.
typedef particle::MortonOrder<Traits> MortonOrder;
// We use Verlet lists to store the neighbors.
typedef particle::VerletLists<MortonOrder> Verlet;
// A number type that is suitable for particle indices.
typedef MortonOrder::Index Index;
// A Cartesian point.
typedef MortonOrder::Point Point;

// The domain for the problem is a unit box.
const geom::BBox<Float, Dimension> Domain =
   {ext::filled_array<Point>(0), ext::filled_array<Point>(1)};
// Construct the data structure for ordering the particles.
MortonOrder mortonOrder(Domain, interactionDistance);
// Construct the data structure for calculating neighbors.
Verlet verlet(mortonOrder);

// Set the initial state of the particles.
{
   std::vector<Particle> particles(numParticles);
   // Initialize the particles.
   ...
   // Set the particles.
   mortonOrder.setParticles(particles.begin(), particles.end());
}

// Take a number of steps.
for (std::size_t step = 0; step != numSteps; ++step) {
   // Repair the data structure if necessary.
   mortonOrder.repair();
   // Calculate the neighbors.
   verlet.findLocalNeighbors();
   // For each particle.
   for (Index i = mortonOrder.localParticlesBegin(); i mortonOrder.localParticlesEnd(); ++i) {
      // Record a reference to the particle for convenience.
      Particle& particle = mortonOrder.particle[i];
      // For each neighbor of the particle.
      for (std::size_t j = 0; j != verlet.neighbors.size(i); ++i) {
         const Neighbor& neighbor = verlet.neighbors(i, j);
         // The neighboring particle's index.
         const Index n = neighbor.particle;
         // The neighboring particle's position.
         Point p = mortonOrder.neighborPosition(neighbor);
         // Do something with the neighbor, like apply forces.
         ...
      }
   }
}
\endcode

\par
If you are familiar with C++ and templates, the source code above is
probably easy to understand. First we define types. The
particle::MortonOrder class that stores and orders the particles is
templated on a traits class. We define traits for a plane domain
with particle::PlainTraits. In the next block we construct
an instance of particle::MortonOrder using the Cartesian domain and
an interaction distance. Then we construct an instance of
particle::VerletLists that we will use for finding neighbors.
The initial state of the particles is set with the
setParticles() member function. With this call, the MortonOrder
class takes ownership of the particles.

\par
After defining types, constructing the particle data structures, and
initializing, we are ready to run the simulation. Each time step starts
with a call to MortonOrder::repair(). The particles will be reordered if
necessary. Next we calculate the neighbors with
VerletLists::findLocalNeighbors(). We access the neighbors
with the packed array VerletLists::neighbors. The
MortonOrder::neighborPosition() function provides a uniform interface
for accessing the postition of neighbor positions. (For periodic domains,
positions may need to be offset.)


<!------------------------------------------------------------------------>
\section particle_highlights Highlights

\par
I'm sure that you are familiar with
<a href="http://en.wikipedia.org/wiki/Quadtree">quadtrees</a>. If not,
follow the link and do a little reading. These data structures are
called quadtrees in 2D and octrees in 3D, Our trees are templated on the
space dimension. Thus, we call them \e orthtrees (short for orthant trees).

\par
We use \e linear orthtrees. For each cell in the tree, we store a
code that identifies it. We will talk
more about these codes later. For now, just accept the fact that the
identity of the cell may be encoded in an unsigned integer. Empty
cells are not stored.
The particles are sorted according to their code values, which
happens to give them good data locality (more about that later). The
data structure for a linear orthtree is simply the vector of sorted
particles along with the vector of cell codes and the vector of
delimiters that define the cells. All of the
data is stored in contiguous arrays. Thus, processing it can be
very efficient. Also, there are no pointers. This is useful for two
reasons. Firstly, it is easy to partition the tree into regions, and
distribute these regions across processes. Secondly, algorithms for
linear orthtrees may easily be ported to architectures that do not
support pointers.

\par
For our linear orthtrees, we only use cells at the highest level of refinement.
We choose that level of refinement based upon the interaction distance
for the particles. In effect, the cells are the right size for the operations
that we will perform. One could find a particular cell in the tree with a binary
search on the codes. The computational complexity of this search is
O(\e n log \e n), where \e n is the number of cells. However, we use
a lookup table (implemented in the LookupTable class) to accelerate searching.

\par
As we mentioned before, the representation of a linear orthree is
remarkably simple. You just store vectors of cell delimiters and codes
along with your
vector of particles. (In our case we also have the lookup table.)
However, the extension to a distributed-memory tree is simpler
still. Each process gets a contiguous range of particles (continuous
in their associated codes), and the particles are globally
sorted. Thus, we can describe the distribution of particles across \e p
processes with an array of \e p+1 delimiters. For an arbitrary cell, a
binary search on the delimiters identifies the process that holds the
cell.

\par
We use the linear orthtree for all of the features that the framework supports.
It is used for ordering, partitioning and exchanging particles, as
well as for finding neighbors. The orthtree is conceptually simple,
has a small storage requirement, and enables efficient computational
geometry algorithms. Instead of being dynamic (updated at each time step),
the orthtree is quasi-static. The cells in the tree are wider than the
interaction distance by a user-specified padding. Repair is only necessary when
any particle has moved more than half of this padding distance.

<!------------------------------------------------------------------------>
\section particle_requirements Requirements

\par
The framework for supporting particle methods is implemented in the
class MortonOrder. For MPI applications, use the derived class
MortonOrderMpi. There are a few requirements for using this
framework, some are mathematical, others constrain the software
implementation. We will consider the former first.
For any simulation, you must define an associated domain that is an
axis-aligned box. (It need not be equilateral.)
For periodic domains, the particles lie
in the box. For plain domains, the particles may be distributed in any
fashion, and may be subject to any boundary conditions, but the box
should contain the particles. We will discuss this further in the \ref
particle_discrete_coordinates "Discrete Coordinates" section.
We assume that particles interact with their neighbors and that the
interaction distance is a constant. While one could use linear orthtrees for
problems whose interaction distance varies in space or time, we
have made this assumption to simplify the implementation and improve
performance. We discuss how the interaction distance affects the orthtree
in the \ref particle_search "Search Distance" section.

\par
The particle must be represented with a C++ struct or class that has a
default constructor, a copy constructor, and an assignment operator.
(In most cases, you would use the synthesized copy constructor and
assignment operator. That is, let the compiler figure out how to copy your
particles class instead of writing the functions yourself. You will need to define a default constructor only if you have defined other constructors.)
These functions are necessary because the MortonOrder class stores the
particles in a
\c std::vector. Thus, the particles are represented using an array of
structures instead of a structure of arrays.
The particle class must be self-contained.
For instance, the particle positions may not
be stored as an external array of points. The framework orders the particles
to achieve good data locality for neighbor calculations, as well as
exchanging particles between sockets in distributed-memory applications.
Thus, for MPI implementations, the particle class
must not reference external data with pointers.
The vector of particles may be reordered and resized. The user does not
have direct control over these operations; they are performed automatically.
Therefore, you may not rely on persistent pointers to particles.
For example, if you store pointers to neighboring particles, you must
recalculate them after the data structure is repaired because
the addresses of particles may
change at that point. (Of course, this is a silly example because
the framework provides access to neighbors. There is no need for redundant
storage.)

\par
The particle class must provide access to its position. For periodic domains,
it must also provide a mechanism for setting the position. When particles
leave the box that defines the domain, the framework will return them
to the box using periodic extension. We will discuss this further in
the \ref particle_positions "Particle Positions" section.


<!------------------------------------------------------------------------>
\section particle_search Search Distance

\par
We assume that the interaction radius is constant across the domain.
Thus, the neighbors for each particle are particles within
a ball of fixed radius. For the sake of efficiency, the cell length is
set to a value that is at least as large as the interaction distance.
This makes it easy to determine the cells
that intersect the search ball. Specifically, in N-D space,
any search ball is contained in 3<sup>N</sup> cells.

\par
While this package could be used for
static problems, it is primarily intended for problems in which the
particles move. It would be wasteful to repair the data structure at
each time step. Thus, we use a padding. Particles are allowed to
move half of the padding distance before repair is necessary.
(If no particle has moved more than half of the padding distance, then
no distance between a pair of particles has changed more than the padding.)
The cell length is at least the sum of
the interaction distance and the padding.
This way it is still easy to determine the cells that intersect the
ball whose radius is the interaction distance.

<!------------------------------------------------------------------------>
\section particle_positions Particle Positions

\par
Consider a sequence of particles. The data structure used to represent
a particle may contain arbitrary information such as velocity and material
properties, but it must be able to provide a position. Of course, we could
define a particle base class that stores a position and then require the user
to inherit from this class. Alternatively, we could require that the
particle class have a particular data member or member function for accessing
the position. However, these designs would be restrictive. Instead, we
only require the user to provide a functor to access the position. Specifically,
the user provides a unary functor whose argument type is a particle, and
whose return type is a point, represented with \c std::array.

\par
For a trivial example, suppose that the only data associated with a particle
is its position. In 3D, one might represent a particle with
\c std::array<float, 3>. Then the position functor is just the identity
function. See ads::Identity for an implementation of this. Below we
define types for the particle and the position accessor.

\par
\code
typedef std::array<float, 3> Particle;
typedef ads::Identity<Particle> GetPosition;
\endcode

\par
For a less trivial example, suppose that the particle stores a position and
velocity as public data members. Below we define the particle class and the
position functor. Both are templated on the space dimension.

\par
\code
template<std::size_t N>
struct Particle {
   typedef std::array<float, N> Point;
   Point position;
   Point velocity;
};

template<std::size_t N>
struct GetPosition :
   public std::unary_function<Particle<N>, typename Particle<N>::Point> {

   typedef std::unary_function<Particle<N>, typename Particle<N>::Point> Base;

   const typename Base::result_type&
   operator()(const typename Base::argument_type& x) const {
      return x.position;
   }
};
\endcode

\par
When defining functors, it is common practice to derive from
\c std::unary_function or its relatives. This bit of boilerplate
automates the process of defining the \c argument_type and
\c result_type. However, for our purposes, it is not necessary.
Below is an alternate definition of the position accessor.

\par
\code
template<std::size_t N>
struct GetPosition {
   const typename Particle<N>::Point&
   operator()(const Particle<N>& x) const {
      return x.position;
   }
};
\endcode

\par
For problems on periodic domains, you will need to provide a functor that sets
the position. The arguments are a pointer to a particle and the point.
Below is an implementation for our example.

\par
\code
template<std::size_t N>
struct SetPosition {
   void
   operator()(Particle<N>* x, const typename Particle<N>::Point& p) const {
      x->position = p;
   }
};
\endcode



\par
Now suppose that in your particle class you don't use \c std::array
to represent the location. (By the way, that's a poor move on your part.
\c std::array is definitely the way to go.) For the sake of the following
example, we'll assume that perhaps you are forced to use legacy code for
the particle or that you're just foolish. Anyway, the following particle
class uses C arrays to represent the position and velocity. And as long as
we are feeling uninspired, we won't template the class on the space dimension.

\par
\code
struct DumbParticle {
   float position[3];
   float velocity[3];
};

struct GetPosition {
   std::array<float, 3>
   operator()(const DumbParticle& x) const {
      std::array<float, 3> p = {{x.position[0], x.position[1], x.position[2]}};
      return p;
   }
};

struct SetPosition {
   void
   operator()(DumbParticle* x, const std::array<float, 3>& p) const {
      x->position[0] = p[0];
      x->position[1] = p[1];
      x->position[2] = p[2];
   }
};
\endcode

<!------------------------------------------------------------------------>
\section particle_discrete_coordinates *Discrete Coordinates

\par
A necessary step in the spatial indexing that we perform is converting
floating-point coordinates to discrete coordinates. The box that defines
the domain of the othtree
is recursively sub-divided a number of times by splitting into
orthants. For example, a 2D box with two levels of refinement has 16
cells. The cells are given zero-offset, integer coordinates with the
axis origin in the lower corner. A diagram of the coordinates is shown below.

\par
<table border="1">
<tr>
<td>(0, 3)</td><td>(1, 3)</td><td>(2, 3)</td><td>(3, 3)</td>
</tr>
<tr>
<td>(0, 2)</td><td>(1, 2)</td><td>(2, 2)</td><td>(3, 2)</td>
</tr>
<tr>
<td>(0, 1)</td><td>(1, 1)</td><td>(2, 1)</td><td>(3, 1)</td>
</tr>
<tr>
<td>(0, 0)</td><td>(1, 0)</td><td>(2, 0)</td><td>(3, 0)</td>
</tr>
</table>

\par
For a given point, floating-point coordinates are converted to
discrete coordinates by determining the cell that contains it.
Usually, the domain is chosen so that all points of interest
are contained, but this is not required. If a point lies outside of the
box, it is simply given the coordinates of the closest cell. However,
for best performance, the box should contain all points of
interest. This way the discrete coordinates are a good approximation
of the actual positions.



\par
The DiscreteCoordinates class is used to calculate discrete coordinates.
When constructing it, along with a domain, one specifies the cell length.
From this, an appropriate number of levels of refinement are determined.
For plain domains, cell lengths are set to the requested size. The level
of refinement is chosen so that the orthtree domain covers the problem
domain. That is, the orthtree domain is equilateral and covers the problem
domain, specifically, it is centered on it. Note that setting the cell lengths
to the requested size is important for performance. The cells are used in
finding neighbors as well as in exchanging particles between sockets in
distributed-memory applications. Using larger cells adversely affect
performance. For finding neighbors, this would increase the number of
potential neighbors that would need to be examined. For exchanging particles,
it would increase the number exchanged.

\par
For the periodic case, one could require that the orthtree domain
match the problem domain. However, this is problematic. The cell lengths
must be at least as long as the requested length. If we could only adjust
the level of refinement, then the cell lengths may be up to twice this length.
In 3D, this means that cells may be as much as eight times larger than the
ideal volume. In addition, requiring that the orthtree domain match the
problem domain is vexing when the latter is not equilateral.
To address this issue, we only require that the lower corners of the
two domains coincide; the orthtree domain may extend beyond the problem
domain. Then, we do not necessarily use the full cell extents.
For example, with three levels of refinement, there may be up to
eight cells in each dimension. However, the cell extents will be less than
or equal to eight.
We only require that the upper boundaries of the domain coincide
with cell boundaries. This allows us to obtain cells that are close
to the ideal size.
Specifically, we choose the largest extent such that
the cell lengths are at least as large as required.



\par
Just to beat this issue to death, let's consider a concrete example in 2D.
Suppose that the input for the domain is the unit box with lower corner
at the origin and that the input for the cell length is 0.2. For a simulation
with a plain domain, there will be three levels of refinement. The cell length
is set to 0.2 and the box is expanded so that the lower corner is at
(-0.3, -0.3) and the upper corner is at (0.3, 0.3). A diagram of the domain
(in red) and the cells (in blue) is shown below.

\image html plain.png "The cells for a plain domain."
\image latex plain.png "The cells for a plain domain."

\par
For a simulation on a periodic domain, the tree will also use three levels
of refinement. The cell extents in each dimension are set to 5. This
again results in a cell length of 0.2. Note that the tree has 8 cells in
each dimension, however, the last 3 are not used. Below we show a diagram
of the domain and the cells. The cells that are not used are outlined in
dashed lines.

\image html periodic.png "The cells for a periodic domain."
\image latex periodic.png "The cells for a periodic domain."

<!------------------------------------------------------------------------>
\section particle_morton *The Morton Code

\par
In particle methods, particles interact with their neighbors. These
computations are more efficient if the particles have good data
locality.  That is, if particles that are close in physical space are
likely to be close in memory. To achieve this, we first associate
particles with the cells in which they lie. This provides a coarse
discretization of the positions.  Then the cell indices are converted
to <a href="http://en.wikipedia.org/wiki/Z-order_curve">Morton
codes</a>. The %Morton order defines a space-filling curve that visits
each of the cells. It defines an ordering of the cells that has good
data locality. (Neighboring cells are likely to be close in memory.)
The %Morton code interleaves the bits in the discrete coordinates
for the cells. First we show the cell coordinates written in as binary numbers.

\par
<table border="1">
<tr>
<td>(00, 11)</td><td>(01, 11)</td><td>(10, 11)</td><td>(11, 11)</td>
</tr>
<tr>
<td>(00, 10)</td><td>(01, 10)</td><td>(10, 10)</td><td>(11, 10)</td>
</tr>
<tr>
<td>(00, 01)</td><td>(01, 01)</td><td>(10, 01)</td><td>(11, 01)</td>
</tr>
<tr>
<td>(00, 00)</td><td>(01, 00)</td><td>(10, 00)</td><td>(11, 00)</td>
</tr>
</table>

The bits are interleaved to produce the %Morton codes. Below they are the
binary codes.

\par
<table border="1">
<tr>
<td>1010</td><td>1011</td><td>1110</td><td>1111</td>
</tr>
<tr>
<td>1000</td><td>1001</td><td>1100</td><td>1101</td>
</tr>
<tr>
<td>0010</td><td>0011</td><td>0110</td><td>0111</td>
</tr>
<tr>
<td>0000</td><td>0001</td><td>0100</td><td>0101</td>
</tr>
</table>

Finally, we show the decimal numbers. This makes it easy to visually
trace the curve.

\par
<table border="1">
<tr>
<td>10</td><td>11</td><td>14</td><td>15</td>
</tr>
<tr>
<td>8</td><td>9</td><td>12</td><td>13</td>
</tr>
<tr>
<td>2</td><td>3</td><td>6</td><td>7</td>
</tr>
<tr>
<td>0</td><td>1</td><td>4</td><td>5</td>
</tr>
</table>

\par
Note that for the ordering to be effective, the particles must be
stored in a contiguous block of memory, in our case a \c std::vector.
The Morton class is used for computing %Morton codes. It inherits from
the DiscreteCoordinates class to get the ability to convert Cartesian
points to cell indices. Like the base class, it is constructed by
specifying the box for the domain and the cell length. It efficiently
converts cell indices to codes by using a data array and working with
eight bits (a byte) at a time.  (The algorithm that works on one bit
at a time is straightforward, but not very efficient.) The inverse
operation uses a similar approach to accelerate the conversion.


<!------------------------------------------------------------------------>
\section particle_indexing *Cell Indexing

\par
With particles and codes ordered together, one can find the particles in any
cell with a binary search. We accelerate this search with the use of a
lookup table, implemented in the LookupTable class. This data structure
has a table for a level of refinement that is less than or equal to that
used for the codes. In the constructor one specifies the maximum allowed
table size. (Typically one would choose a size that is on the order of
the number of particles.) If the level of refinement for the table matches
that for the codes, then lookup is performed in constant time. For lower
levels of refinement, lookups are used to obtain a range for a subsequent
binary search.


<!------------------------------------------------------------------------>
\section particle_serial Serial and Shared-Memory Applications

\par Traits.
You specify the properties of your simulation with a traits class. These
define the following.
- particle class
- position accessor
- position manipulator (for periodic domains)
- whether the domain is periodic
- space dimension
- floating-point number type
.
The Traits class has a template parameter for each of these. In most cases,
it will be convenient to use either PlainTraits or PeriodicTraits, depending
on whether your domain is plain or periodic. Both of these have a shorter
list of template parameters. For PlainTraits you must specify the particle
class and the functor for accessing particle positions. Optionally, you
may specify the problem dimension (the default is 3) and the
floating-point number type (the default is single-precision).

\par Constructor.
For serial or shared-memory applications, we use the MortonOrder class.
It is templated on a traits class.
Construct it by specifying
the domain for your problem (an axis-aligned bounding box), the
interaction distance, and (optionally) the padding.
(The default value of the padding is usually fine.)
In the code example below,
we use a plain (non-periodic) domain.
We assume that the \c Particle class and the \c GetPosition functor have
been suitably defined.

\par
\code
typedef particle::PlainTraits<Particle, GetPosition> Traits;
typedef particle::MortonOrder<Traits> MortonOrder;
const geom::BBox<float, 3> Domain = {{{0, 0, 0}}, {{1, 1 ,1}}};
const float InteractionDistance = 0.1;
MortonOrder mortonOrder(Domain, InteractionDistance);
\endcode

\par
The class also has a default constructor and an initialize() function for
achieving the same effect.

\par
\code
MortonOrder mortonOrder;
mortonOrder.initialize(Domain, InteractionDistance);
\endcode

\par
For problems on periodic domains, you will need to provide a functor
for setting particle positions.

\par
\code
typedef particle::PeriodicTraits<Particle, GetPosition, SetPosition> Traits;
typedef particle::MortonOrder<Traits> MortonOrder;
MortonOrder mortonOrder(Domain, InteractionDistance);
\endcode

\par Set particles.
Use the MortonOrder::setParticles() function to set the particles.
The codes will be computed, and the relevant data structures will be
initialized. Below is a simple example.

\par
\code
// A particle is just a point.
typedef std::array<float, 3> Particle;
// Make a vector of 1000 particles.
std::vector<Particle> particles(1000);
// Initialize the particles.
...
// Set the particles.
mortonOrder.setParticles(particles.begin(), particles.end());
\endcode

\par
The particles are stored in a \c std::vector and
are accessible through the public data member MortonOrder::particles.
For an example, below is code for printing the particles.
(Here we assume that the particle class has a suitably defined
\c %operator<<(), either as a member or a free function.)

\par
\code
for (std::size_t i = 0; i != mortonOrder.particles.size(); ++i) {
   std::cout << mortonOrder.particles[i] << '\n';
}
\endcode

\par Accessors.
You can access the interaction distance, squared interaction distance,
and the padding with the member
functions MortonOrder::interactionDistance(),
MortonOrder::squaredInteractionDistance(), and MortonOrder::padding(),
respectively. MortonOrder::localParticlesBegin() returns the index of the
first local particle, while MortonOrder::localParticlesEnd() return one past
the index of the last local particle. For serial and shared-memory
applications, all particles are local. These functions are only
provided for compatibility with the distributed-memory interface.

\par
Recall that you can access the particles through the
MortonOrder::particles public member. They are ordered according to the
%Morton order, with the provision that a particle is within half of the padding
distance of the cell. Note that allowing particles to wander a small
distance outside of their associated cells is critical.
Requiring particles to always be contained in the
cells, would be prohibitively expensive. In practice, it would
necessitate repairing the data structure at every time step.
For plain domains, the user has full authority to ignore this detail.
The underlying tree data structure is hidden from the user anyway.
However, for periodic domains, this has definite consequences. Particles are
\e not guaranteed to be in the specified domain. They may be up to half the
padding distance outside of it. Thus, the user must not assume that the
particles are in the domain. For force calculations, this is not a problem.
Below, we will see how neighbor positions are reported relative to a given
particle. However, one might have a visualization routine, for example,
that shows the particle positions in the periodic domain. Here, the
positions would need to be corrected.

\par Repair.
At the beginning of each time step,
you must call MortonOrder::repair(). If
the data structure actually needs repair, the codes will be recalculated and
the particles will be reordered. If the repair
is done, the function returns true.
Otherwise, it does nothing and returns false.
It is necessary to check the return value only if you have auxiliary data
structures that depend on the order of the particles.

\par Neighbors.
For periodic domains, the effective position of a neighbor may be a
virtual periodic
extension of the actual stored position. To make this clear, consider a
trivial 1D case with a domain of unit length and an interaction distance
of 0.1. A particle at 0.01 might have a neighbor at
0.99 because -0.01 is a periodic extension of that position. When working
with neighbors (applying forces or the like) one needs to use the
location that makes the particle a neighbor.

\par
The particle::Neighbor struct is used to represent neighbors. It is
templated on a Boolean value that indicates whether the domain is
periodic.
For plain domains, particle::Neighbor<false> holds a particle index
in the \c particle member. For periodic domains, particle::Neighbor<true>
also holds an index that defines a periodic offset in the
\c offset member. The \c particle member may be used directly to
get the index of a neighbor particle. The \c offset member is used
indirectly. The MortonOrder::neighborPosition() function, which
takes a particle::Neighbor as an argument, provides a
uniform interface for accessing the position of a neighbor particle.
For plain domains it is simply a wrapper for the MortonOrder::position()
function, but for periodic domains, it uses the \c offset member to
add an appropriate offset to the position.

\par
There are a number of classes that may be used to compute neighbors.
The simplest (but also the least efficient) is VerletLists.
We will cover its use first.
All of the neighbor classes store a const reference to the
MortonOrder data structure and take this as an argument in their
constructor.

\par
\code
typedef particle::VerletLists<MortonOrder> Verlet;
Verlet verlet(mortonOrder);
\endcode

\par
At each time step, you will need to calculate the neighbors with either
VerletLists::findLocalNeighbors() or VerletLists::findAllNeighbors().
The former calculates neighbors for the local particles. The latter
also calculates neighbors for the shadow particles. For serial and
shared-memory applications, the functions produce the same result
because there are on shadow particles.
You can access the neighbors through the public VerletLists::neighbors
member. This is a packed array of \c particle::Neighbor.
Below we verify the validity
of the reported neighbors. We check that the index of the neighbor is
different than that of the source particle and that the position is
within the interaction distance. Note that we allow for round-off errors.

\par
\code
typedef Verlet::Neighbor Neighbor;
verlet.findLocalNeighbors();
const float Eps = Length * std::numeric_limits<float>::epsilon();
for (Index i = 0; i != mortonOrder.particles.size(); ++i) {
   for (std::size_t j = 0; j != verlet.neighbors.size(i); ++j) {
      const Neighbor& neighbor = verlet.neighbors(i, j);
      assert(i != neighbor.particle);
      assert(euclideanDistance(mortonOrder.position(i),
                               mortonOrder.neighborPosition(neighbor)) <=
             mortonOrder.interactionDistance() + Eps);
   }
}
\endcode

\par Shared-memory architectures.
If you compile with OpenMP enabled, MortonOrder will use threading when
repairing the data structure. Thus, you must call MortonOrder::repair()
in a serial block. For your part, to take advantage of multi-core
architectures, you just need to
partition the particles amongst the threads. This
could be as simple as inserting a pragma before your loop over the
particles. Note that below, we use a signed integer for the iteration
variable to avoid a compilation warning.

\par
\code
bool repaired;
for (std::size_t step = 0; step != NumSteps; ++step) {
   mortonOrder.repair();
   verlet.findLocalNeighbors();
   ...
   // Take a time step.
#pragma omp parallel for default(none)
   for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(mortonOrder.particles.size()); ++i) {
      // Calculate forces and move the i_th particle.
      ...
   }
}
\endcode


<!------------------------------------------------------------------------>
\section particle_distributed_applications Distributed-Memory Applications


\par Overview.
For distributed-memory applications, use the MortonOrderMpi class. It has
the same template parameters and nearly the same interface as its
parent class MortonOrder.

\par
First a bit of terminology: we call computational units that share memory
\e sockets. Sockets are composed of cores that are capable of running a
certain number of threads. As we saw in the previous section, sharing work
amongst the cores in a socket is trivial; each thread is responsible for
an equal portion of the particles.
To utilize distributed-memory architectures, the cells (and their
associated particles) are distributed
amongst the sockets. Load-balancing is an important issue.
We use a fairly simple approach. For each cell, the cost function is
the number of
potential neighbors. This is the product of the number of particles in the
cell and the number in the adjacent cells. The computational costs for the
most expensive parts of a simulation scale with the number of neighbors.
Thus, the potential neighbor counts are roughly proportional to the
costs associated with a cell.
Each socket has a roughly equal share of the cell costs.
The particles are divided in such a way that each
socket holds a contiguous, disjoint range of %Morton codes. Thus, we can
specify the partitioning among \e N sockets in terms of <i>N+1</i>
delimiters. The delimiters are %Morton codes. The nth socket holds
the cells whose codes are at least as large as the nth delimiter
and less than the (n+1)th delimiter.

\par
To illustrate the partitioning we show a 2D example below. The cells are
distributed among eight sockets, identified by different colors.
Tracing the %Morton curve, we see that
each socket holds a contiguous range of cells.

\image html partitionConnected.png "The components may be connected."
\image latex partitionConnected.png "The components may be connected."

\par
Of course, the partition above is only one possibility. Below we show another.
Note that not all of the components are connected. However, the components are
compact. That is, their boundary is small compared to their content.

\image html partitionCompact.png "Although not conected, the components are compact."
\image latex partitionCompact.png "Although not conected, the components are compact."

\par
In addition to the local particles, each socket must also hold a copy of
their potential neighbors that reside on other sockets. Because the partition
respects the %Morton ordering, distinguishing between local and foreign
particles is very easy. The local particles are a contiguous block
in the particles array. The accessors MortonOrder::localParticlesBegin()
and MortonOrder::localParticlesEnd() define their index range. So when you
work with the particles, instead of iterating over all of them,
iterate over the local range. To port your shared-memory application
to a distributed memory one, you replace
\code
for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(mortonOrder.particles.size()); ++i) {
\endcode
with the following.
\code
for (std::ptrdiff_t i = mortonOrder.localParticlesBegin(); i < std::ptrdiff_t(mortonOrder.localParticlesEnd()); ++i) {
\endcode
Of course, if you are crafty, you will use the latter idiom for your
shared-memory code as well since those functions are defined in
MortonOrder.

\par
The foreign particles occupy blocks
that precede and follow the local range. This convention is handy
because two indices discriminate between local and foreign. There is
no need to store that information in the particle class itself.

\par Constructor.
The constructors for MortonOrderMpi differ from that of MortonOrder
in that they take an MPI intra-communicator as the first argument.
Below is an example of constructing the class for a plain domain.

\par
\code
const geom::BBox<float, 3> Domain = {{{0, 0, 0}}, {{1, 1 ,1}}};
const float InteractionDistance = 0.1;
particle::MortonOrderMpi<Traits> mortonOrder(MPI::COMM_WORLD, Domain, InteractionDistance);
\endcode

\par Set particles and repair.
Setting the particles and repairing the data structure has the same
interface as before. Just use MortonOrderMpi::setParticles()
and MortonOrderMpi::repair(). However, these functions do a little
more work than their counterparts in MortonOrder. In setting
the particles, they are distributed amongst the sockets. While all
sockets must call the MortonOrderMpi::setParticles() function, you are
free to introduce particles in any fashion that you like. One socket or
all sockets may add particles. The particles may have any initial positions.
After distributing the particles, the pattern for exchanging potential
neighbors is calculated, and then the potential neighbors are exchanged.
Upon completion of MortonOrderMpi::setParticles() the data structure
is ready for the first time step.

\par
In addition to possibly reordering the particles, MortonOrderMpi::repair()
may re-partition them to balance the load. If it does either, it
needs to determine a new exchange pattern. If not, it will reuse the pattern
and just exchange particles.

<!------------------------------------------------------------------------>
\section particle_distributed_algorithms *Distributed-Memory Algorithms

\par Load Balancing.
Through load-balancing, we ensure that each socket has an approximately
equal estimated simulation cost. The load imbalance is calculated by
determining the maximum cost, dividing by the
mean, and then subtracting one. The load imbalance
is an estimate of how much more work the most heavily loaded process does
compared to the average. The maximum allowed imbalance is stored in the
MortonOrder::maxLoadImbalance member data. The default value is 0.01,
but you are free to set it to a different value.

\par
\code
// Set the maximum allowed load imbalance to 5%.
mortonOrder.maxLoadImbalance = 0.05;
\endcode

\par
Note that in distributing the load, it is cells that are distributed, and not
individual particles. That is, the partitioning defines the range of
cells (with associated %Morton codes) that each process owns. Thus, depending
on the distribution of particles and the maximum allowed load imbalance,
it may not be possible to generate a partition that is sufficiently balanced.
This situation might arise from a particular cell having many particles, or
could result from the user setting \c maxLoadImbalance to a very small value.
In order to avoid the possibility of performing an expensive partitioning
at every time step, we record the starting load imbalance every time we
calculate a partitioning. We will repartition only if the current imbalance
exceeds \c maxLoadImbalance \e and it exceeds twice the product of a
scaling factor and the starting imbalance. (The scaling factor is initially
1 and is reduced by 1\% at each time step.)


<!------------------------------------------------------------------------>
\section particle_serial_performance Serial Performance

\par
In this section we will consider the performance of various operations on
linear orthtrees. We will consider some basic operations like manipulating
%Morton codes as well as more sophisticated ones like finding neighbors.
The tests were conducted on an Apple Macbook Pro with a
2.8 GHz Intel Core 2 Duo processor and 8 GB of 1067 MHz DDR3 RAM.

\par Floating-point operations.
Before considering the orthtree, we present timing information for
some basic floating-point operations. Below is a table of the time
(in nanoseconds) that is required for a single operation or function
call. These results will give us a reference point for
evaluating the rest of the timings.
<table border="1">
<tr><th>Operation</th> <th>Time (ns)</th>
<tr><td>Multiplication</td><td>3.3</td>
<tr><td>Division</td><td>7.2</td>
<tr><td>Square root</td><td>7.2</td>
<tr><td>Exponential</td><td>14.2</td>
<tr><td>Logarithm</td><td>14.9</td>
</table>

\par
First we will consider basic operations with the %Morton code. We will conduct
the tests using 3D orthtrees with varying numbers of levels. Since
we use 64-bit integers to store the codes, a 3D orthtree may have
up to 21 levels of refinement.

\par Discrete coordinates to %Morton code.
Below are timing results for converting discrete coordinates in 3D to a %Morton
code. The process consists of interleaving the bits.

\image html coordinatesToCode.jpg "Discrete Coordinates to Morton Code."
\image latex coordinatesToCode.pdf "Discrete Coordinates to Morton Code."

\par
We see that the cost of the operation rises with increasing levels of
refinement, but it is inexpensive. Note that we employ an algorithm
that works with 8 bits (a byte) at a time. This is much more efficient
than doing one at a time. This also explains the plateaus and jumps
in the timing results.


\par %Morton code to discrete coordinates.
Next we consider the inverse operation. We convert a %Morton code to
discrete coordinates in 3D.
code.

\image html codeToCoordinates.jpg "Morton Code to Discrete Coordinates."
\image latex codeToCoordinates.pdf "Morton Code to Discrete Coordinates."

\par
We see that this operation is also inexpensive. This is because we have
implemented it with an efficient algorithm. In 3D, it processes 9 bits
of the code at a time. Thus we see jumps in the cost at levels that
are multiples of three.


\par Cartesian point to %Morton code.
Next we convert a Cartesian location in 3D to a %Morton code.
This is performed when assigning a particle to a cell. The operation is
done in two steps. The Cartesian coordinates are converted to discrete
coordinates and then this is converted to the %Morton code.
Below we see that this is a reasonably inexpensive operation.

\image html cartesianToCode.jpg "Cartesian Point to Morton Code."
\image latex cartesianToCode.pdf "Cartesian Point to Morton Code."


\par
In the remaining tests, we will work with a set of particles that
are uniformly, randomly distributed in a unit cube. In the tests,
we will vary the number of particles.


\par Cell lookup by %Morton code, ordered.
We consider finding the cell that contains particles with a
specified code. This is an essential operation that is used in many
orthtree algorithms. For instance, it is used when finding the neighboring
particles. For each of the particles in the orthtree, we look up
the cell that contains particles with that code. First we do the queries
in order, ascending order for the %Morton codes, that is.
Below we show the number of particles, the number of levels of refinement
in the tree, the shift amount for the lookup table, and the
time to find the cell in nanoseconds. We see that in all cases,
the lookup is very fast.
<table border="1">
<tr><th>Particles</th> <th>Levels</th> <th>Shift</th> <th>Time (ns)</th>
<tr><td>1000</td><td>8</td><td>13</td><td>8</td>
<tr><td>1000</td><td>12</td><td>25</td><td>7</td>
<tr><td>1000</td><td>16</td><td>37</td><td>7</td>
<tr><td>1000</td><td>20</td><td>49</td><td>6</td>
<tr><td>10000</td><td>8</td><td>9</td><td>6</td>
<tr><td>10000</td><td>12</td><td>21</td><td>5</td>
<tr><td>10000</td><td>16</td><td>33</td><td>12</td>
<tr><td>10000</td><td>20</td><td>45</td><td>8</td>
<tr><td>100000</td><td>8</td><td>6</td><td>7</td>
<tr><td>100000</td><td>12</td><td>18</td><td>7</td>
<tr><td>100000</td><td>16</td><td>30</td><td>8</td>
<tr><td>100000</td><td>20</td><td>42</td><td>11</td>
<tr><td>1000000</td><td>8</td><td>3</td><td>7</td>
<tr><td>1000000</td><td>12</td><td>15</td><td>7</td>
<tr><td>1000000</td><td>16</td><td>27</td><td>7</td>
<tr><td>1000000</td><td>20</td><td>39</td><td>7</td>
</table>

\par Cell lookup by %Morton code, random.
Next we repeat the above test, but do the queries in a random order.
For up to 100,000 particles, the performance is a little slower than
before, but still very fast. However, the lookups with 1,000,000 particles
are much expensive. This is because the particles and lookup table will no
longer fit in the L2 cache. The random nature of the queries causes cache
misses. Note that many of the orthtree algorithms schedule the queries
so that they are approximately ordered. This reduces cache misses, even
when working with many particles.
<table border="1">
<tr><th>Particles</th> <th>Levels</th> <th>Shift</th> <th>Time (ns)</th>
<tr><td>1000</td><td>8</td><td>13</td><td>8</td>
<tr><td>1000</td><td>12</td><td>25</td><td>8</td>
<tr><td>1000</td><td>16</td><td>37</td><td>7</td>
<tr><td>1000</td><td>20</td><td>49</td><td>7</td>
<tr><td>10000</td><td>8</td><td>9</td><td>7</td>
<tr><td>10000</td><td>12</td><td>21</td><td>12</td>
<tr><td>10000</td><td>16</td><td>33</td><td>11</td>
<tr><td>10000</td><td>20</td><td>45</td><td>8</td>
<tr><td>100000</td><td>8</td><td>6</td><td>16</td>
<tr><td>100000</td><td>12</td><td>18</td><td>15</td>
<tr><td>100000</td><td>16</td><td>30</td><td>14</td>
<tr><td>100000</td><td>20</td><td>42</td><td>14</td>
<tr><td>1000000</td><td>8</td><td>3</td><td>54</td>
<tr><td>1000000</td><td>12</td><td>15</td><td>55</td>
<tr><td>1000000</td><td>16</td><td>27</td><td>55</td>
<tr><td>1000000</td><td>20</td><td>39</td><td>55</td>
</table>


\par Order the particles.
Below we show the cost of ordering the particles. For this,
the %Morton codes are computed and then the particles are sorted
by these values. The times are measured in nanoseconds per particle.
<table border="1">
<tr><th>Particles</th> <th>Levels</th> <th>Time (ns)</th>
<tr><td>1000</td><td>8</td><td>95</td>
<tr><td>1000</td><td>12</td><td>87</td>
<tr><td>1000</td><td>16</td><td>88</td>
<tr><td>1000</td><td>20</td><td>104</td>
<tr><td>10000</td><td>8</td><td>109</td>
<tr><td>10000</td><td>12</td><td>110</td>
<tr><td>10000</td><td>16</td><td>109</td>
<tr><td>10000</td><td>20</td><td>105</td>
<tr><td>100000</td><td>8</td><td>128</td>
<tr><td>100000</td><td>12</td><td>123</td>
<tr><td>100000</td><td>16</td><td>123</td>
<tr><td>100000</td><td>20</td><td>127</td>
<tr><td>1000000</td><td>8</td><td>200</td>
<tr><td>1000000</td><td>12</td><td>198</td>
<tr><td>1000000</td><td>16</td><td>196</td>
<tr><td>1000000</td><td>20</td><td>204</td>
</table>
We see that the number of levels of refinement has little effect.
The performance primarily depends on the number of particles.
The cost is certainly much more than for the basic operations
we previously considered, but it is still reasonable.


\par Finding neighbors.
Finally, we present timing information for finding neighbors. We vary the
number of particles, and the search radius. For each test we report
the average number of neighbors per particle and the average time
per reported neighbor in nanoseconds.
<table border="1">
<tr><th>Particles</th> <th>Radius</th> <th>Neighbors</th> <th>Time (ns)</th>
<tr><td>1000</td><td>0.10</td><td>3.6</td><td>256</td>
<tr><td>1000</td><td>0.20</td><td>25.5</td><td>61</td>
<tr><td>1000</td><td>0.40</td><td>158.1</td><td>29</td>
<tr><td>10000</td><td>0.046</td><td>3.9</td><td>216</td>
<tr><td>10000</td><td>0.093</td><td>30.1</td><td>57</td>
<tr><td>10000</td><td>0.19</td><td>215.8</td><td>37</td>
<tr><td>100000</td><td>0.022</td><td>4.1</td><td>208</td>
<tr><td>100000</td><td>0.043</td><td>32.0</td><td>58</td>
<tr><td>100000</td><td>0.086</td><td>243.0</td><td>42</td>
<tr><td>1000000</td><td>0.0100</td><td>4.1</td><td>212</td>
<tr><td>1000000</td><td>0.020</td><td>32.8</td><td>59</td>
<tr><td>1000000</td><td>0.040</td><td>256.2</td><td>44</td>
</table>
Note that the cost per neighbor is high when there are few neighbors.
For the tests in which each particle has about four neighbors, one does
a lot of expensive searching for few neighbors. When each
particles has tens or hundreds of neighbors, the cost is quite
modest. Furthermore, the cost scales well with the number of particles.

<!------------------------------------------------------------------------>
\section particle_shared_performance Shared-Memory Performance

\par
We run a simulation that performs trivial operations at each time step.
(The program source is <tt>test/performance/particle/openmp/random.cc</tt>.)
Specifically, it counts the neighbors and moves each particle
according to a random vector. We run on SHC using the dual-processor,
quad-core nodes.  We perform a strong scaling study, holding constant
the number of particles while increasing the number of cores. There
are 100,000 particles, and the simulation takes 1000 steps. The command
is shown below where <tt>T</tt> is the number of threads.
<pre>
./random -p=1000000 -s=1000 -m=periodic -t=T
</pre>

\par
The interaction distance is such that each particle has
about 30 neighbors. At every time step, each particle moves 1% of
the interaction distance
in a random direction. With 1,000 steps, the particles are reordered
15 times.

\par
We measure the time spent in simulating the particle motion and in
repairing the orthtree data structure. The former has two components
which we label <tt>MoveParticles</tt> and <tt>CountNeighbors</tt>.
Moving the particles simply consists of adding a random vector to the
position. In any particle simulation, each step at least has nested loops
over the particles and over their neighbors. Ordinarily, one would use
the positions and attributes to compute forces. Here, we merely count
the total number of neighbors. (Recall that the data structure stores
\em potential neighbors. At each time step, the distances must
be computed to determine which potential neighbors are actually neighbors.
The distance must be computed anyway to calculate forces, so storing
potential neighbors is efficient.) Anyway, the costs for simulating
particle motion in this trivial application are lower than for any
real physical problem.

\par
Now we consider the costs of repairing the orthtree data structure.
There are five components, which we label and describe below.
- <tt>CheckOrder</tt>. Check how far the particle positions have moved from
their initial positions (when the data structure was last repaired). If any
particle has moved further than half the padding, repair is necessary.
This operation is done at every time step. The remaining operations only
occur when the data structure is repaired.
- <tt>Order</tt>. Calculate %Morton codes for the particles and then
sort them according to those codes.
- <tt>RecStartPos</tt>. Record the starting positions of the particles.
- <tt>%LookupTable</tt>. Build the lookup table that is used to accelerate
searching for particles by their %Morton codes.
- <tt>FindNeighbors</tt>. Determine the potential neighbors for each
particle. Store them in a packed array.

\par
In the figure below, we show a chart of the total time for each operation,
for simulations using between 1 and 8 cores. First note that the costs
of repairing the orthtree data structure are small compared to the
other costs. Specifically, counting the number of neighbors is the
dominant cost. We see that the total simulation time decreases as we
increase the core count.

\image html randomSharedTimeAll.png "Total Time for All Operations."
\image latex randomSharedTimeAll.png "Total Time for All Operations."

\par
Next we scale all of the times, multiplying by the number of cores.
The figure below shows the scalability of the operations. A constant cost
with increasing core count is perfect scalability. We see that moving the
particles (<tt>MoveParticles</tt>) scales very well. This is no surprise,
it is trivial to make the operation concurrent, and the calculation is
compute-bound with regular memory access. Counting neighbors also has nearly
perfect scalability. While this operation is again trivial to make
concurrent, the memory access is complex. Also, we are doing very little
calculation with each neighbor. Specifically, we are only computing
distance to count the
actual neighbors. Here, the orthtree sufficiently regularizes the memory
access to keep the operation compute-bound as we increase the number of cores.

\image html randomSharedScaledAll.png "Scaled Time for All Operations."
\image latex randomSharedScaledAll.png "Scaled Time for All Operations."

\par
Since the costs of repairing the orthtree are small compared to
the other operations, we will examine these costs separately. In the
figure below, we show the total costs for the five operations. We see that
finding neighbors is the most expensive part, and that the cost of this
operation decreases with increasing core count. A distant second is
checking whether the data structure needs repair (<tt>CheckOrder</tt>).
This operation utilizes threading; we can see the cost decrease with
increasing core count.

\image html randomSharedTimeRepair.png "Total Time for Repair Operations."
\image latex randomSharedTimeRepair.png "Total Time for Repair Operations."

\par
Next we examine the scaled timings. We see that finding potential
neighbors does not scale as well as the other simulation costs we
considered above. Specifically, in going from one core to eight, the
cost is only reduced by a factor of 4.4, an efficiency of 55%. There are a
couple of reasons for this. Obviously, not all of the aspects of finding
neighbors can be made concurrent. While the particles may partitioned for
finding neighbors, the result must be assembled into one packed array.
Thus, there are memory allocation operations that are inherently serial.
But more importantly, the operation is memory-bound. There is a lot of memory
access for little arithmetic calculation. While each thread acts
independently, progress is likely throttled by the overall memory access
capabilities of the socket.

\image html randomSharedScaledRepair.png "Scaled Time for Repair Operations."
\image latex randomSharedScaledRepair.png "Scaled Time for Repair Operations."

<!------------------------------------------------------------------------>
\section particle_concurrent_performance Concurrent Performance

\par
We run a simulation that performs trivial operations at each time step.
(The program source is <tt>test/performance/particle/mpi/random.cc</tt>.)
Specifically, it counts the neighbors and moves each particle according
to a random vector. We run on SHC using the dual-processor, dual-core nodes.
We perform a weak scaling study, increasing the number of particles with
the number of nodes. We use MPI for communication between nodes, and
OpenMP for concurrency within a node.
There are 100,000 particles per node, and the simulation
takes 1,000 steps.
The command
for running the simulation is shown below where <tt>N</tt> is the number
of nodes.
<pre>
mpirun -np N ./random -p=100000 -s=1000 -m=periodic
</pre>

\par
The interaction distance is such that each particle has about 30
neighbors. At every time step, each particle moves 1% of the
interaction distance in a random direction. With 1,000 steps, the
particles are reordered between 13 and 16 times, depending on the
process count. This is sufficient to compare the costs of repairing
the orthtree (which occurs when any particle has moved more than half
the padding distance) with exchanging particles, moving them, and
accessing their neighbors (which are performed at each time step). Of
course, the repair costs and the running costs depend on the
simulation characteristics. Problems in which the distribution of
particles changes rapidly will have higher repair costs.  Problems in
which particles have many neighbors will have higher running costs.

\par
We measure the time spent in simulating the particle motion and in
repairing the orthtree data structure. The former has two components
which we again label <tt>MoveParticles</tt> and <tt>CountNeighbors</tt>.
These represent the same operations as before.

\par
Now we consider the costs of building and repairing the orthtree data
structure.
There are six components, which we label and describe below.
- <tt>Reorder</tt>. Recalculate %Morton codes and sort the particles.
Move particles to the appropriate processes.
- <tt>Partition</tt>. Determine the partitioning of the codes so that
particles will be evenly distributed across processes.
- <tt>Distribute</tt>. After determining the partition, move each particle
to the process to which it belongs.
- <tt>ExPattern</tt>. Determine the pattern that will be used for exchanging
particles at each time step.
- <tt>Exchange</tt>. Exchange particles between the processes. This updates
the foreign particles for each process.
- <tt>Neighbors</tt>. Determine the potential neighbors for each
particle. Store them in a packed array.

\par
The <tt>Partition</tt> and <tt>Distribute</tt> operations are carried
out when building the data structure and whenever the load becomes imbalanced.
A <tt>Reorder</tt> occurs when any particle has moved more than half the padding
distance from its position at the last reordering. This necessitates a
recalculation of the exchange pattern (<tt>ExPattern</tt>) and
determining the potential neighbor (<tt>Neighbors</tt>).
The <tt>Exchange</tt> operation is done at each time step.

\par
In the figure below, we show a chart of the total time for each operation,
for simulations using between 4 and 128 cores (1 and 32 nodes).
First note that the costs
of building and repairing the orthtree data structure are less than the
costs of moving the particles and counting the number of neighbors.
For a simulation that actually does something with the neighbors (like
calculating forces with the neighbors instead of just counting them),
the gap would be
much wider. We see that the costs associated with moving the particles
and accessing their neighbors scales well as the process count increases.
Although there is some variability, there is no upward trend. This is as
one would expect. If the load is balanced, these simulation costs should
not increase.

\image html randomMpi4TimeAll.png "Total Time for All Operations."
\image latex randomMpi4TimeAll.png "Total Time for All Operations."

\par
Next we plot only the costs associated with building and repairing
the orthtree. Starting at four processes (16 cores), exchanging particles,
which is done at each time step, becomes the most expensive task.
There is significant variation in the cost as the process count increases,
but no clear trend. Determining the potential neighbors, which is done after
reordering, is the second most expensive operation. This remains fairly
constant as the process count increases.

\image html randomMpi4TimeRepair.png "Total Time for Repair Operations."
\image latex randomMpi4TimeRepair.png "Total Time for Repair Operations."

<!--
\image html timeTotals.jpg "Time for Orthree Operations."
\image latex timeTotals.pdf "Time for Orthree Operations."

\image html occupancy.jpg "Cell Occupancy."
\image latex occupancy.pdf "Cell Occupancy."

\image html exchangeCount.jpg "Number of Particles Exchanged."
\image latex exchangeCount.pdf "Number of Particles Exchanged."
-->

\section particle_bibliography Bibliography

<!--The paper that introduces linear quadtrees.-->
<!--
@article{ ISI:A1982PW19900006,
Author = {GARGANTINI, I},
Title = {{AN EFFECTIVE WAY TO REPRESENT QUADTREES}},
Journal = {{COMMUNICATIONS OF THE ACM}},
Year = {{1982}},
Volume = {{25}},
Number = {{12}},
Pages = {{905-910}},
DOI = {{10.1145/358728.358741}},
ISSN = {{0001-0782}},
Unique-ID = {{ISI:A1982PW19900006}},
}
-->
-# \anchor particle_gargantini1982
I. Gargantini. "An Effective Way to Represent Quadtrees."
Communications of the ACM, Vol. 25, No. 12, 1982.

<!-- Interleave bits to calculate %Morton codes.
http://graphics.stanford.edu/~seander/bithacks.html#InterleaveTableLookup
-->

<!-- Cannot obtain through Caltech connect.
Title: EFFICIENT SECONDARY MEMORY PROCESSING OF WINDOW QUERIES ON SPATIAL DATA
Author(s): NARDELLI, E; PROIETTI, G
Source: INFORMATION SCIENCES  Volume: 84   Issue: 1-2   Pages: 67-83   DOI: 10.1016/0020-0255(94)00107-M   Published: MAY 1995
Times Cited: 6 (from Web of Science)
-->

<!------------------------------------------------------------------------>
<!--
\section particle_links Links
-->

<!------------------------------------------------------------------------>
<!--
\section particle_todo To Do

- Consider Hilbert codes.
-
-->

*/
}
}

/*
$ qsub -I -l nodes=1:core4 -l walltime=00:05:00
qsub: waiting for job 424298.mistress to start
qsub: job 424298.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 4 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.000849195
Partition count = 1
Reorder count = 1
Average time spent in various functions:
Reorder: 0.0822062
Partition: 0.006109
Distribute: 0.368326
Exchange Pattern: 0.0885193
Exchange Particles: 0.188738
Find Neighbors: 0.525679


$ qsub -I -l nodes=2:core4 -l walltime=00:05:00
qsub: waiting for job 424299.mistress to start
qsub: job 424299.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 8 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00153367
Partition count = 1
Reorder count = 1
Average time spent in various functions:
Reorder: 0.0816399
Partition: 0.012798
Distribute: 0.489485
Exchange Pattern: 0.121103
Exchange Particles: 0.25687
Find Neighbors: 0.52889


$ qsub -I -l nodes=4:core4 -l walltime=00:05:00
qsub: waiting for job 424300.mistress to start
qsub: job 424300.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 16 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00400805
Partition count = 1
Reorder count = 1
Average time spent in various functions:
Reorder: 0.0849734
Partition: 0.0153526
Distribute: 0.527503
Exchange Pattern: 0.165416
Exchange Particles: 0.707856
Find Neighbors: 0.624928


$ qsub -I -l nodes=8:core4 -l walltime=00:05:00
qsub: waiting for job 424301.mistress to start
qsub: job 424301.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 32 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00415447
Partition count = 1
Reorder count = 1
Average time spent in various functions:
Reorder: 0.0880657
Partition: 0.0207735
Distribute: 0.680056
Exchange Pattern: 0.185898
Exchange Particles: 0.775963
Find Neighbors: 0.624494


$ qsub -I -l nodes=16:core4 -l walltime=00:05:00
qsub: waiting for job 424302.mistress to start
qsub: job 424302.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 64 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00414349
Partition count = 1
Reorder count = 1
Average time spent in various functions:
Reorder: 0.0921553
Partition: 0.0279421
Distribute: 1.47885
Exchange Pattern: 0.140849
Exchange Particles: 1.02216
Find Neighbors: 0.613327


$ qsub -I -l nodes=32:core4 -l walltime=00:05:00
qsub: waiting for job 424303.mistress to start
qsub: job 424303.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 128 ./random -p=100000 -s=100
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00323573
Partition count = 1
Reorder count = 2
Average time spent in various functions:
Reorder: 0.192551
Partition: 0.0459537
Distribute: 2.24806
Exchange Pattern: 0.421203
Exchange Particles: 2.26988
Find Neighbors: 1.3803

 */

/*
-------------------------------------------------------------------------------
-bash-3.2$ qsub -I -l nodes=1:core4 -l walltime=00:05:00
qsub: waiting for job 424375.mistress to start
qsub: job 424375.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 4 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.000849195
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0837065,0.00307675,0.590187,0.0993973,0.531904,2.68557e+06,2.68557e+06,1.03655
Min,0.080929,0.002766,0.583623,0.095204,0.530156,2.66582e+06,2.67662e+06,1.03144
Max,0.08541,0.003369,0.596976,0.106629,0.533631,2.70141e+06,2.69245e+06,1.04012

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0837065,0.00307675,0.590187,0.0496986,0.518274
Min,0.080929,0.002766,0.583623,0.047602,0.515721
Max,0.08541,0.003369,0.596976,0.0533145,0.520062

Per step:
,X Particles,X Sent,X Received
Average,0.00531904,26855.7,26855.7
Min,0.00530156,26658.2,26766.2
Max,0.00533631,27014.1,26924.5

Move particles per step = 0.00837291
Interactions per particle per step = 9.99597
Count neighbors per step = 0.0532049


-bash-3.2$ qsub -I -l nodes=2:core4 -l walltime=00:05:00
qsub: waiting for job 424376.mistress to start
qsub: job 424376.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 8 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00153367
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0847897,0.00357537,0.900859,0.087348,0.787347,4.27832e+06,4.27832e+06,1.71381
Min,0.081329,0.002967,0.891453,0.071582,0.759996,4.23921e+06,4.25477e+06,1.69542
Max,0.088207,0.004276,0.913853,0.100325,0.804698,4.30482e+06,4.30583e+06,1.75801

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0847897,0.00357537,0.900859,0.043674,0.856906
Min,0.081329,0.002967,0.891453,0.035791,0.847708
Max,0.088207,0.004276,0.913853,0.0501625,0.879005

Per step:
,X Particles,X Sent,X Received
Average,0.00787347,42783.2,42783.2
Min,0.00759996,42392.1,42547.7
Max,0.00804698,43048.2,43058.3

Move particles per step = 0.00836182
Interactions per particle per step = 9.9984
Count neighbors per step = 0.0540924


-bash-3.2$ qsub -I -l nodes=4:core4 -l walltime=00:05:00
qsub: waiting for job 424377.mistress to start
qsub: job 424377.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 16 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.0020571
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0989435,0.0356181,0.647318,0.169457,0.582361,2.72157e+06,2.72157e+06,0.712993
Min,0.093911,0.010601,0.637795,0.159034,0.55398,2.69788e+06,2.68559e+06,0.704058
Max,0.103165,0.046484,0.653935,0.177832,0.612714,2.75599e+06,2.73586e+06,0.764326

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0989435,0.0356181,0.647318,0.0847287,0.356496
Min,0.093911,0.010601,0.637795,0.079517,0.352029
Max,0.103165,0.046484,0.653935,0.088916,0.382163

Per step:
,X Particles,X Sent,X Received
Average,0.00582361,27215.7,27215.7
Min,0.0055398,26978.8,26855.9
Max,0.00612714,27559.9,27358.6

Move particles per step = 0.00839333
Interactions per particle per step = 9.9952
Count neighbors per step = 0.0531791


-bash-3.2$ qsub -I -l nodes=8:core4 -l walltime=00:05:00
qsub: waiting for job 424378.mistress to start
qsub: job 424378.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 32 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00224744
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.104211,0.0524084,0.863212,0.207855,1.05817,3.48179e+06,3.48179e+06,1.04115
Min,0.098311,0.009666,0.850595,0.159671,0.832723,3.44573e+06,3.4517e+06,1.02399
Max,0.110257,0.05807,0.916082,0.224696,1.28975,3.53945e+06,3.5222e+06,1.09223

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.104211,0.0524084,0.863212,0.103928,0.520574
Min,0.098311,0.009666,0.850595,0.0798355,0.511993
Max,0.110257,0.05807,0.916082,0.112348,0.546115

Per step:
,X Particles,X Sent,X Received
Average,0.0105817,34817.9,34817.9
Min,0.00832723,34457.3,34517
Max,0.0128975,35394.5,35222

Move particles per step = 0.00839706
Interactions per particle per step = 10.004
Count neighbors per step = 0.0530187


-bash-3.2$ qsub -I -l nodes=16:core4 -l walltime=00:05:00
qsub: waiting for job 424379.mistress to start
qsub: job 424379.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 64 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00229136
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0952225,0.0614986,1.85155,0.215712,1.47411,4.27888e+06,4.27888e+06,1.70661
Min,0.088633,0.01199,1.82773,0.167301,1.24994,4.22665e+06,4.24593e+06,1.68309
Max,0.103079,0.070186,1.91906,0.264105,1.71743,4.32528e+06,4.32216e+06,1.77747

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0952225,0.0614986,1.85155,0.107856,0.853304
Min,0.088633,0.01199,1.82773,0.0836505,0.841543
Max,0.103079,0.070186,1.91906,0.132052,0.888734

Per step:
,X Particles,X Sent,X Received
Average,0.0147411,42788.8,42788.8
Min,0.0124994,42266.5,42459.3
Max,0.0171743,43252.8,43221.6

Move particles per step = 0.00837858
Interactions per particle per step = 10.0009
Count neighbors per step = 0.054759


-bash-3.2$ qsub -I -l nodes=32:core4 -l walltime=00:05:00
qsub: waiting for job 424380.mistress to start
qsub: job 424380.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 128 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00280015
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.135167,0.0596682,2.66107,0.290138,1.23488,2.72742e+06,2.72742e+06,0.804199
Min,0.126972,0.021345,2.62778,0.234159,0.881986,2.67107e+06,2.68607e+06,0.753366
Max,0.144033,0.069912,2.71202,0.338086,1.54997,2.77401e+06,2.76224e+06,0.920254

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.135167,0.0596682,2.66107,0.145069,0.4021
Min,0.126972,0.021345,2.62778,0.11708,0.376683
Max,0.144033,0.069912,2.71202,0.169043,0.460127

Per step:
,X Particles,X Sent,X Received
Average,0.0123488,27274.2,27274.2
Min,0.00881986,26710.7,26860.7
Max,0.0154997,27740.1,27622.4

Move particles per step = 0.00841209
Interactions per particle per step = 9.99664
Count neighbors per step = 0.0541808
 */

/*
------------------------------------------------------------------------------
-bash-3.2$ qsub -I -l nodes=1:core8 -l walltime=00:05:00
qsub: waiting for job 424382.mistress to start
qsub: job 424382.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 8 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00153367
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0752974,0.002634,0.777841,0.0538101,0.601234,4.27832e+06,4.27832e+06,1.56013
Min,0.07362,0.001923,0.769267,0.04668,0.598397,4.23921e+06,4.25477e+06,1.54333
Max,0.077137,0.004303,0.78345,0.063895,0.604224,4.30482e+06,4.30583e+06,1.58658

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0752974,0.002634,0.777841,0.0269051,0.780066
Min,0.07362,0.001923,0.769267,0.02334,0.771667
Max,0.077137,0.004303,0.78345,0.0319475,0.793288

Per step:
,X Particles,X Sent,X Received
Average,0.00601234,42783.2,42783.2
Min,0.00598397,42392.1,42547.7
Max,0.00604224,43048.2,43058.3

Move particles per step = 0.00798552
Interactions per particle per step = 9.9984
Count neighbors per step = 0.0502818


-bash-3.2$ qsub -I -l nodes=2:core8 -l walltime=00:05:00
qsub: waiting for job 424383.mistress to start
qsub: job 424383.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 16 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.0020571
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0751836,0.008,0.465709,0.0983676,0.458616,2.72157e+06,2.72157e+06,0.661381
Min,0.072388,0.006667,0.454769,0.092534,0.437833,2.69788e+06,2.68559e+06,0.654963
Max,0.07725,0.009607,0.471283,0.11019,0.483193,2.75599e+06,2.73586e+06,0.667337

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0751836,0.008,0.465709,0.0491838,0.330691
Min,0.072388,0.006667,0.454769,0.046267,0.327481
Max,0.07725,0.009607,0.471283,0.055095,0.333669

Per step:
,X Particles,X Sent,X Received
Average,0.00458616,27215.7,27215.7
Min,0.00437833,26978.8,26855.9
Max,0.00483193,27559.9,27358.6

Move particles per step = 0.00834656
Interactions per particle per step = 9.9952
Count neighbors per step = 0.0494052


-bash-3.2$ qsub -I -l nodes=4:core8 -l walltime=00:05:00
qsub: waiting for job 424384.mistress to start
qsub: job 424384.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 32 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00224744
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0758438,0.0320814,0.914106,0.0875055,0.884193,3.48179e+06,3.48179e+06,0.953618
Min,0.068861,0.007066,0.898978,0.073593,0.795236,3.44573e+06,3.4517e+06,0.940525
Max,0.080757,0.04102,0.924151,0.10581,0.942192,3.53945e+06,3.5222e+06,0.975277

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0758438,0.0320814,0.914106,0.0437528,0.476809
Min,0.068861,0.007066,0.898978,0.0367965,0.470263
Max,0.080757,0.04102,0.924151,0.052905,0.487639

Per step:
,X Particles,X Sent,X Received
Average,0.00884193,34817.9,34817.9
Min,0.00795236,34457.3,34517
Max,0.00942192,35394.5,35222

Move particles per step = 0.0080084
Interactions per particle per step = 10.004
Count neighbors per step = 0.0497916


-bash-3.2$ qsub -I -l nodes=8:core8 -l walltime=00:05:00
qsub: waiting for job 424385.mistress to start
qsub: job 424385.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 64 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00229136
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0752367,0.0420194,1.78402,0.153031,1.46155,4.27888e+06,4.27888e+06,1.55026
Min,0.067431,0.007028,1.76929,0.095228,1.38264,4.22665e+06,4.24593e+06,1.53108
Max,0.08096,0.04565,1.80575,0.197758,1.5663,4.32528e+06,4.32216e+06,1.57296

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0752367,0.0420194,1.78402,0.0765153,0.775128
Min,0.067431,0.007028,1.76929,0.047614,0.76554
Max,0.08096,0.04565,1.80575,0.098879,0.786478

Per step:
,X Particles,X Sent,X Received
Average,0.0146155,42788.8,42788.8
Min,0.0138264,42266.5,42459.3
Max,0.015663,43252.8,43221.6

Move particles per step = 0.0079693
Interactions per particle per step = 10.0009
Count neighbors per step = 0.0506175


-bash-3.2$ qsub -I -l nodes=16:core8 -l walltime=00:05:00
qsub: waiting for job 424386.mistress to start
qsub: job 424386.mistress ready

-bash-3.2$ cd Development/stlib/test/performance/release/particle/mpi/
-bash-3.2$ mpirun -np 128 ./random -p=100000 -s=100 -m=periodic
_Dimension = 3, numParticles = 100000
Starting imbalance = 0.00280015
Partition count = 1
Reorder count = 1
Repair count = 100

Totals:
,Reorder,Partition,Distribute,X Pattern,X Particles,X Sent,X Received,Neighbors
Average,0.0774668,0.0478448,2.32737,0.178119,0.860052,2.72742e+06,2.72742e+06,0.7404
Min,0.06961,0.012737,2.3116,0.153139,0.636354,2.67107e+06,2.68607e+06,0.697008
Max,0.083425,0.058498,2.34641,0.202526,1.01998,2.77401e+06,2.76224e+06,0.830779

Per operation:
,Reorder,Partition,Distribute,X Pattern,Neighbors
Average,0.0774668,0.0478448,2.32737,0.0890594,0.3702
Min,0.06961,0.012737,2.3116,0.0765695,0.348504
Max,0.083425,0.058498,2.34641,0.101263,0.415389

Per step:
,X Particles,X Sent,X Received
Average,0.00860052,27274.2,27274.2
Min,0.00636354,26710.7,26860.7
Max,0.0101998,27740.1,27622.4

Move particles per step = 0.00785748
Interactions per particle per step = 9.99664
Count neighbors per step = 0.0487547

 */
/*
Using subcell resolution for sorting the particles increases the effective
operations when using bitmask methods. The amount of improvement depends
on the number of neighbors per particle. When the number of neighbors is small,
so is the improvement. It seems best to use 2 levels of subcell sorting.
There is diminishing return in using more, even when the number of neighbors is
fairly large.

./randomAdjacent -p=10000 -n=60 -s=1 -m=periodic
Neighbor density = 0.0756783

No subcell sorting:
Nonzero mask dens. = 0.244007, eff. dens. = 0.310148, eff. ops. = 2.48119.
Nonzero nibble dens. = 0.209517, eff. dens. = 0.361203, eff. ops. = 1.44481.

1 level of subcell sorting:
Nonzero mask dens. = 0.204444, eff. dens. = 0.370166, eff. ops. = 2.96133.
Nonzero nibble dens. = 0.174782, eff. dens. = 0.432986, eff. ops. = 1.73195.

2 levels of subcell sorting:
Nonzero mask dens. = 0.20164, eff. dens. = 0.375314, eff. ops. = 3.00251.
Nonzero nibble dens. = 0.172272, eff. dens. = 0.439294, eff. ops. = 1.75718.
Updated. Array ordering:
Nonzero mask dens. = 0.199117, eff. dens. = 0.380069, eff. ops. = 3.04055.
Nonzero nibble dens. = 0.171564, eff. dens. = 0.441109, eff. ops. = 1.76444.


./randomAdjacent -p=10000 -n=200 -s=1 -m=periodic
Neighbor density = 0.0473348

No subcell sorting:
Nonzero mask dens. = 0.164703, eff. dens. = 0.287394, eff. ops. = 2.29915.
Nonzero nibble dens. = 0.141638, eff. dens. = 0.334196, eff. ops. = 1.33678.

1 level of subcell sorting:
Nonzero mask dens. = 0.108341, eff. dens. = 0.436905, eff. ops. = 3.49524.
Nonzero nibble dens. = 0.0965689, eff. dens. = 0.490166, eff. ops. = 1.96066.

2 levels of subcell sorting:
Nonzero mask dens. = 0.0992591, eff. dens. = 0.476881, eff. ops. = 3.81505.
Nonzero nibble dens. = 0.0879835, eff. dens. = 0.537996, eff. ops. = 2.15198.

3 levels of subcell sorting:
Nonzero mask dens. = 0.0984559, eff. dens. = 0.480771, eff. ops. = 3.84617.
Nonzero nibble dens. = 0.0872355, eff. dens. = 0.542609, eff. ops. = 2.17044.
*/
#endif

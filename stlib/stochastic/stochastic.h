// -*- C++ -*-

#if !defined(__stochastic_h__)
#define __stochastic_h__

#include "stlib/stochastic/Direct.h"
#include "stlib/stochastic/EssTerminationCondition.h"
#include "stlib/stochastic/FirstReaction.h"
#include "stlib/stochastic/FirstReactionAbsoluteTime.h"
#include "stlib/stochastic/FirstReactionInfluence.h"
#include "stlib/stochastic/HistogramFinalHypoexponentialDirect.h"
#include "stlib/stochastic/HistogramReference.h"
#include "stlib/stochastic/HistogramsAverage.h"
#include "stlib/stochastic/HistogramsAverageAps.h"
#include "stlib/stochastic/HistogramsAverageBase.h"
//#include "stlib/stochastic/HistogramsAverageElapsedMultiTime.h"
#include "stlib/stochastic/HistogramsAverageElapsedTime.h"
//#include "stlib/stochastic/HistogramsAverageMultiTime.h"
#include "stlib/stochastic/HistogramsAveragePacked.h"
#include "stlib/stochastic/HistogramsAveragePackedArray.h"
#include "stlib/stochastic/HistogramsAveragePackedDouble.h"
#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/HistogramsDirect.h"
#include "stlib/stochastic/HistogramsDirectTree.h"
#include "stlib/stochastic/HistogramsMultiTimeDirect.h"
#include "stlib/stochastic/HistogramsPacked.h"
#include "stlib/stochastic/HistogramsPackedArray.h"
#include "stlib/stochastic/HistogramsPackedDouble.h"
#include "stlib/stochastic/HomogeneousHistogramsTransientDirectTreeExponentialLast.h"
#include "stlib/stochastic/HomogeneousHistogramsTransientDirectTreeExponentialLimit.h"
#include "stlib/stochastic/HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit.h"
#include "stlib/stochastic/HomogeneousHistogramsTransientDirectTreeNormalApproximation.h"
#include "stlib/stochastic/HybridDirectTauLeaping.h"
#include "stlib/stochastic/InhomogeneousDirect.h"
#include "stlib/stochastic/InhomogeneousHistogramsSteadyStateDirect.h"
#include "stlib/stochastic/InhomogeneousHistogramsTransientDirect.h"
#include "stlib/stochastic/InhomogeneousTimeSeriesAllReactionsDirect.h"
#include "stlib/stochastic/InhomogeneousTimeSeriesUniformDirect.h"
#include "stlib/stochastic/NextReaction.h"
#include "stlib/stochastic/OdeReaction.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/PropensitiesInhomogeneous.h"
#include "stlib/stochastic/PropensityTimeDerivatives.h"
#include "stlib/stochastic/Reaction.h"
#include "stlib/stochastic/ReactionPriorityQueue.h"
#include "stlib/stochastic/ReactionSet.h"
#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/State.h"
#include "stlib/stochastic/TauLeaping.h"
#include "stlib/stochastic/TauLeapingDynamic.h"
#include "stlib/stochastic/TauLeapingImplicit.h"
#include "stlib/stochastic/TauLeapingSal.h"
#include "stlib/stochastic/TerminationCondition.h"
#include "stlib/stochastic/TimeEpochOffset.h"
#include "stlib/stochastic/TrajectoryTreeFull.h"
// No longer supported.
//#include "stlib/stochastic/api.h"
#include "stlib/stochastic/modifiedRecordedSpecies.h"
#include "stlib/stochastic/reactionPropensityInfluence.h"

// CONTINUE
// Hybrid method with direct method and equilibrium. Partition the reactions
// into slow and fast. At each step drive the fast reactions to equilibrium
// and then take a step with the direct method. If the fast reactions cannot
// be driven to equilibrium in a time that is short compared to the expected
// direct time step, then some reactions must be moved to the slow group.
// Closed reaction sets that are dominating the slow group are candidates
// to be moved to the fast group.
// Perhaps the user can enter the time scale of interest.

// CONTINUE
// Hybrid direct/implicit ODE. This might be better than the ssSSA approach
// for some problems. Use the populations to partition.

/*
  BioSpice: Does not work on the Mac. There is an error on startup. I cannot
  install modules from the web. I tried manually downloading and installing
  ESS (Exact Stochastic Simulations), but the module did not load.
*/

namespace stlib
{
//! All classes and functions in the stochastic package are defined in the stochastic namespace.
namespace stochastic {}
}

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\mainpage Stochastic Simulations for a Set of Reactions

I have investigated a few topics in %stochastic simulations for chemical
kinetics. I have been working on
\ref stochastic_exact "efficient algorithms for exact simulations".
I have some adaptations of Gillespie's direct method and Gibson and
Bruck's next reaction method that have better computation complexity
and better performance than previous work.

As \ref numerical_random "random number generation" is important in
%stochastic simulations, I have implemented a package with uniform and
non-uniform generators. It has quite a few
of the recently developed algorithms. I've worked on improving the
performance of Poisson generators (useful in tau-leaping) and have developed
some new discrete, finite generators (useful in the direct method).

Previously I looked at
\ref stochastic_concurrent "concurrent algorithms for tau-leaping".
Although concurrent algorithms may be formulated, they are not practical for
generating a suite of trajectories on multi-core computers or networks of
computers.

<!-------------------End \mainpage Stochastic Simulations-------------------->
*/




/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page stochastic_exact Exact Simulations


<!--------------------------------------------------------------------------->
\section stochastic_introduction Introduction

Consider a system of \e N species represented by the state vector
\f$X(t) = (X_1(t), \ldots X_N(t))\f$.  \f$X_n(t)\f$ is the population of the
\f$n^{\mathrm{th}}\f$ species at time \e t.  There are \e M reaction channels
which change the state of the system.  Each reaction is characterized by
a propensity function \f$a_m\f$ and a state change vector
\f$V_m = (V_{m1}, \ldots, V_{mN})\f$.  \f$a_m \mathrm{d}t\f$ is the
probability that the \f$m^{\mathrm{th}}\f$ reaction will occur in the
infinitessimal time interval \f$[t .. t + \mathrm{d}t)\f$.  The state
change vector is the difference between the state after the reaction and
before the reaction.

To generate a trajectory (a possible realization of the evolution of the
system) one starts with an initial state and then repeatedly fires reactions.
To fire a reaction, one must answer the two questions:
-# When will the next reaction fire?
-# Which reaction will fire next?
.
Let the next reaction have index \f$\mu\f$ and fire at time \f$t + \tau\f$.
Let \f$\alpha\f$ be the sum of the propensities.  The time to the
next reaction is an exponentially distributed random variable with mean
\f$1 / \alpha\f$,
\f[
P(\tau = x) = \alpha \mathrm{e}^{-\alpha x}.
\f]
The index of the next reaction to fire is a discrete, finite random variable
with probability mass function \f$P(\mu = m) = a_m / \alpha\f$.


<!--------------------------------------------------------------------------->
\section stochastic_exact Exact Simulations



<!--------------------------------------------------------------------------->
\subsection stochastic_exact_direct The Direct Method

Once the state vector \e X has been initialized, Gillespie's direct
method proceeds by
repeatedly stepping forward in time until a termination condition is reached.
(See \ref stochastic_gillespie1977 "Exact Stochastic Simulation of Coupled Chemical Reactions.")
At each step, one generates two uniform random deviates in the interval
(0..1).  The first deviate, along with the sum of the propensities,
is used to generate an exponential deviate which
is the time to the first reaction to fire.  The second deviate is used to
determine which reaction will fire.
Below is the algorithm for a single step.

-# for m in [1..M]:
  - Compute \f$a_m\f$ from \e X.
-# \f$\alpha = \sum_{m = 1}^M a_m(X)\f$
-# Generate two unit, uniform random numbers \f$r_1\f$ and \f$r_2\f$.
-# \f$\tau = - \ln(r_1) / \alpha\f$
-# Set \f$\mu\f$ to the minimum index such that
\f$\sum_{m = 1}^{\mu} a_m > r_2 \alpha\f$
-# \f$t = t + \tau\f$
-# \f$X = X + V_{\mu}\f$

Consider the computational complexity of the direct method.
We assume that the reactions are loosely coupled and hence
computing a propensity \f$a_m\f$ is \f$\mathcal{O}(1)\f$.
(For tightly coupled reactions, this cost could be as high as
\f$\mathcal{O}(M)\f$.)  Thus the cost of computing the propensities
is \f$\mathcal{O}(M)\f$. Determining \f$\mu\f$ requires iterating over
the array of propensities and thus has cost \f$\mathcal{O}(M)\f$.
With our loosely coupled assumption, updating the state has unit cost.
Therefore the computational complexity of a step with the direct method is
\f$\mathcal{O}(M)\f$.


<!--------------------------------------------------------------------------->
\subsection stochastic_exact_first The First Reaction Method

Gillespie's first reaction method generates a uniform random deviate for
each reation at each time step.  These uniform deviates are used to compute
exponential deviates which are the times at which each reation will next fire.
By selecting the minimum of these times, one identifies the time and
the index of the first reaction to fire.  The algorithm for a single step
is given below.

-# for m in [1..M]:
  - Compute \f$a_m\f$ from \e X.
  - Generate a unit, uniform random number \e r.
  - \f$\tau_m = \ln(1/r) / a_m\f$
-# \f$\tau = \min_m \tau_m\f$
-# \f$\mu =\f$ index of the minimum \f$\tau_m\f$
-# \f$t = t + \tau\f$
-# \f$X = X + V_{\mu}\f$




<!--------------------------------------------------------------------------->
\subsection stochastic_exact_next The Next Reaction Method

Gibson and Bruck's next reaction method is an adaptation of the first
reaction method.
(See \ref stochastic_gibson2000 "Efficient Exact Stochastic Simulation of Chemical Systems with Many Species and Many Channels.")
Instead of computing the time to each reaction, one deals
with the time at which a reaction will occur.  These times are not computed
anew at each time step, but re-used.  The reaction times are stored in an
indexed priority queue (\e indexed because the reaction indices are stored
with the reaction times).  Also, propensities are computed only
when they have changed.  Below is the algorithm for a single step.

-# Get the reaction index \f$\mu\f$ and the reaction time \f$\tau\f$ by
   removing the minimum element from the priority queue.
-# \f$t = \tau\f$
-# \f$X = X + V_{\mu}\f$
-# For each propensity m (\f$i \neq \mu\f$) that is affected by
   reaction \f$\mu\f$:
  - \f$\alpha = \f$ updated propensity.
  - \f$\tau_m = (a_m / \alpha)(\tau_m - t) + t\f$
  - \f$\a_m = \alpha\f$
  - Update the priority queue with the new value of \f$\tau_m\f$
-# Generate an exponential random variable \e r with mean \f$a_{\mu}\f$.
-# \f$\tau_m = t + r\f$
-# Push \f$\tau_m\f$ into the priority queue.


Consider the computational complexity of the next reaction method.
We assume that the reactions are loosely coupled and hence
computing a propensity \f$a_m\f$ is \f$\mathcal{O}(1)\f$.
Let \e D be an upper bound on the number of propensities that are affected by
firing a single reaction.  Then the cost of updating the propensities
and the reaction times is \f$\mathcal{O}(D)\f$. Since the cost of
inserting or changing a value in the priority queue is
\f$\mathcal{O}(\log M)\f$, the cost of updating the priority queue
is \f$\mathcal{O}(D \log M)\f$.
Therefore the computational complexity of a step with the next
reaction  method is \f$\mathcal{O}(D \log M)\f$.


<!--------------------------------------------------------------------------->
\section stochastic_direct An Efficient Formulation of the Direct Method

To improve the computational complexity of the direct method, we first write
it in a more generic way. Initialize the state \e X, the propensities
\f$a_m\f$ and their sum \f$\alpha\f$.  A time step consists of the following:


-# \f$\tau = \mathrm{exponentialDeviate()} / \alpha\f$
-# \f$\mu\f$ = discreteDeviate()
-# \f$t = t + \tau\f$
-# \f$X = X + V_{\mu}\f$
-# Update the discrete, finite deviate generator.
-# Update the sum of the propensities \f$\alpha\f$

There are several ways of improving the performance of the direct method:
- Use a faster algorithms to generate exponential deviates and discrete,
finite deviates.
- Use sparse arrays for the state change vectors.
- Continuously update \f$\alpha\f$ instead of recomputing it at each time step.

The original formulation of the direct method uses the inversion method to
generate an exponential deviate. This is easy to program, but is
computationally expensive due to the evaluation of the logarithm. There are
a couple of recent algorithms (ziggurat and acceptance complement) that have
much better performance. See the \ref numerical_random_exponential for details.

There are many algorithms for generating discrete, finite
deviates. The static case (fixed probability mass function) is well
studied. The simplest approach is CDF inversion with a linear
search. One can implement this with a build-up or chop-down search on
the PMF. The method is easy to code and does not require storing the
CDF. However, it has linear complexity in the number of events, so it
is quite slow.  A better approach is CDF inversion with a binary
search. For this method, one needs to store the CDF.  The binary
search results in logarithmic computational complexity.  A better
approach still is Walker's algorithm, which has constant complexity.
Walker's algorithm is a binning approach in which each bin represents
either one or two events.

Generating discrete, finite deviates with a dynamically changing PMF
is significantly trickier than in the static case. CDF inversion with
a linear search adapts well to the dynamic case; it does not have any
auxiliary data structures. The faster methods have significant
preprocessing costs.  In the dynamic case these costs are incurred in
updating the PMF. The binary search and Walker's algorithm both have
linear preprocessing costs.  Thus all three considered algorithms have
the same complexity for the combined task of generating a deviate and
modifying the PMF. There are algorithms that can both efficiently generate
deviates and modify the PMF.  In fact, there is a method that has constant
complexity. See \ref numerical_random_discrete for details.

The original formulation of the direct method uses CDF inversion with a
linear search. Subsequent versions have stored the PMF in sorted order or
used CDF inversion with a binary search. These modifications have yielded
better performance, but have not changed the worst-case computational
complexity of the algorithm. Using a more sophisticated discrete, finite
deviate generator will improve the performance of the direct method,
particularly for large problems.

For representing reactions and the state change vectors, one can use
either \ref ads::Array "dense" or \ref ads_array_SparseArray1 "sparse"
arrays. Using dense arrays is more efficient for small or tightly
coupled problems. Otherwise sparse arrays will yield better
performance. Consider loosely coupled problems.  For small problems
one can expect modest performance benefits (10 %) in using dense
arrays. For more than about 30 species, it is better to use sparse
arrays.

For loosely coupled problems, it is better to continuously update the
sum of the propensities \f$\alpha\f$ instead of recomputing it at each
time step.  Note that this requires some care. One must account for
round-off error and periodically recompute the sum. Also one must pay
attention to the special case \f$\alpha = 0\f$.

\subsection stochastic_direct_performance Performance

Death, uniformly distributed propensities.
\htmlinclude DirectDeathUniform.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude DirectImmigrationDeath.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude DirectImmigrationDeath100.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude DirectImmigrationDeath1000.txt

Decaying-dimerizing.
\htmlinclude DirectDecayingDimerizing.txt

Auto regulatory.
\htmlinclude DirectAutoRegulatory.txt


<!--------------------------------------------------------------------------->
\section stochastic_next An Efficient Formulation of the Next Reaction Method

One can reformulate the next reaction method to obtain a more efficient
algorithm.  The most expensive parts of the algorithm are maintaining
the binary heap, updating the state, and generating exponential deviates.
Improving the generation of exponential deviates is a minimally invasive
procedure.  Instead of using the inversion method, one can use the
Zigurrat method or the acceptance complement method.
(See \ref stochastic_marsaglia2000 "The ziggurat method for generating random variables"
and
\ref stochastic_rubin2006 "Efficient generation of exponential and normal deviates.")
For updating the state,
one uses the state change vectors.  Except for small problems
(few species), using sparse arrays will yield better performance than
using dense arrays.  Using sparse arrays is a standard technique.
However, this problem has a special structure which enables one
further optimization.  Since one always updates the same state vector,
one can store pointers into that vector instead of array indices.
This means replacing array indexing with pointer dereferencing when
updating the state.  Reducing the cost of the binary heap operations is
a more complicated affair.  We present several approaches below.

\subsection stochastic_next_indexed Indexed Priority Queues

The term <em>priority queue</em> has almost become synonymous with
<em>binary heap</em>.
For most applications, a binary heap is an efficient way of implementing
a priority queue.  For a heap with \e M elements, one can access the minimum
element in constant time. The cost to insert or extract an element or
to change the value
of an element is \f$\mathcal{O}(\log M)\f$.  Also, the storage requirements
are linear in the number of elements.  While a binary heap is rarely the
most efficient data structure for a particular application, it is usually
efficient enough.  If performance is important and the heap operations
constitute a significant portion of the computational cost in an application,
then it may be profitable to consider other data structures.

\subsubsection stochastic_next_indexed_linear Linear Search

The simplest method of implementing a priority queue is to store the elements
in an array and use a linear search to find the minimum element.
The computational complexity of finding the minimum element is
\f$\mathcal{O}(M)\f$.  Inserting, deleting, and modifying elements can be
done in constant time.  For the next reaction method, linear search is
the most efficient algorithm when the number of reactions is small.
It may also be the most efficient method for larger numbers of reactions
that are tightly coupled.  For tightly coupled problems, the dominant cost
is changing the element values.  For linear search, changing an element value
is a simple matter of modifying an array element.

Surprisingly, there are quite a few ways of coding a linear search.
An unguarded search is almost always more efficient than a guarded search.
After that, the performance of a particular method often depends upon
the compiler and the hardware. Sometimes manual loop unrolling pays off.
(Automatic loop unrolling is usually not an issue as most compilers will
not unroll loops containing branches.)
Sometimes searching on an array of pointers into the data is significantly
faster than searching directly on the data array.
(Although it does not directly deal with linear searching, consult
\ref stochastic_Goedecker2001 "Performance Optimization of Numerically Intensive Codes." for an introduction to some of these issues.)


\subsubsection stochastic_next_indexed_partition Partitioning

For larger problem sizes, one can utilize the under-appreciated method
of partitioning.  One stores the elements in an array, but classifies the
the elements into two categories: \e lower and \e upper.  One uses a splitting
value to discriminate; the elements in the lower partition are less than
the splittng value.  Then one can determine the minimum value in the queue
with a linear search on the elements in the lower partition.  Inserting,
erasing, and modifying values can all be done in constant time.  However,
there is the overhead of determining in which partition an element belongs.
When the lower partition becomes empty, one must choose a new splitting
value and re-partition the elements (at cost \f$\mathcal{O}(M)\f$).
By choosing the splitting value so that there are \f$\mathcal{O}(\sqrt{M})\f$
elements in the lower partition, one can attain an average const of
\f$\mathcal{O}(\sqrt{M})\f$ for determining the minimum element.
This choice balances the costs of searching and re-partitioning.
The cost of a search, \f$\mathcal{O}(\sqrt{M})\f$, times the number
of searches before one needs to re-partition, \f$\mathcal{O}(\sqrt{M})\f$,
has the same complexity as the cost of re-partitioning.  There are
several strategies for choosing the splitting value and partitioning
the elements.  Partitioning with a linear search is an efficient method
for problems of moderate size.

\subsubsection stochastic_next_indexed_binary Binary Heaps

When using indexed binary heaps, there are a few implementation details
that have a significant impact on performance.  Gibson and Bruck's
implementation uses one array that holds pairs of reaction indices and reaction
times.  An additional array holds pointers into the first array such that
the \f$n^{\mathrm{th}}\f$ element points to the pair with reaction index
<em>n</em>.  This is not a particularly efficient data structure for
an indexed binary heap.  Inserting, erasing, and modifying elements
are all accomplished by swapping elements in the queue.  In this case,
one swaps pairs of reaction indices and reaction times.  A better approach
is to use three arrays.  The first array holds the reaction times,
the \f$n^{\mathrm{th}}\f$ element is the reaction time for the
\f$n^{\mathrm{th}}\f$ reaction.  The second array is the binary heap,
which stores pointers into the first array.  The third array holds indices
which are the positions of each element in the queue.  This data structure
reduces the cost of swapping.

The algorithm used for modifying elements has a significant impact on
performance.  Gibson and Bruck's algorithm uses tail recursion.  This
is usually not a problem for modern optimizing compilers, it will be
transformed to
a loop.  However, their algorithm uses more branching statements than are
necessary. By using an efficient data structure and updating algorithm,
one can typically reduce the computational costs by at least 1/3.

Binary heaps have decent performance for a wide range of problem sizes.
Because the algorithms are fairly simple, they perform well for small
problems.  Because of the logarithmic complexity, they are suitable for
fairly large problems.


\subsubsection stochastic_next_indexed_hashing Hashing

There is a data structure that can perform each of the operations
(finding the minimum element, inserting, removing, and modifying)
in constant time.  This is accomplished with hashing. (One could also
refer to the method as bucketing.)  The reaction times are stored in
a hash table.
(See \ref stochastic_cormen2001 "Introduction to Algorithms, Second Edition.")
The hashing function is a linear function of the reaction
time (with a truncation to convert from a floating point value to an
integer index).
The constant in the linear function is chosen to give the desired load.
For hashing with chaining, if the load is \f$\mathcal{O}(1)\f$, then all
operations can be done in constant time.  As with binary heaps, the
implementation is important.  Using a generic data structure, such as
\c std::list or \c std::vector for the chaining container will not
yield the best performance.  A special-purpose container can reduce
the overhead in maintaining the hash table.

See the \ref ads_indexedPriorityQueue "indexed priority queues" page
for performance results.

\subsection stochastic_next_performance Performance

Death, uniformly distributed propensities.
\htmlinclude NextReactionDeathUniform.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude NextReactionImmigrationDeath.txt

Decaying Dimerizing
\htmlinclude NextReactionDecayingDimerizing.txt

Auto regulatory.
\htmlinclude NextReactionAutoRegulatory.txt

<!--------------------------------------------------------------------------->
\section stochastic_first An Efficient Formulation of the First Reaction Method

Death, uniformly distributed propensities.
\htmlinclude FirstReactionDeathUniform.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude FirstReactionImmigrationDeath.txt

Decaying-dimerizing.
\htmlinclude FirstReactionDecayingDimerizing.txt

Auto regulatory.
\htmlinclude FirstReactionAutoRegulatory.txt

<!--------------------------------------------------------------------------->
\section stochastic_compare Performance Comparison

Death, uniformly distributed propensities.
\htmlinclude CompareDeathUniform.txt

Immigration-death, uniformly distributed propensities.
\htmlinclude CompareImmigrationDeath.txt

Decaying-dimerizing.
\htmlinclude CompareDecayingDimerizing.txt

Auto regulatory.
\htmlinclude CompareAutoRegulatory.txt


<!-------------------End \page stochastic_exact Exact Simulations------------>
*/





/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page stochastic_concurrent Concurrent Tau-Leaping

<!--------------------------------------------------------------------------->
\section stochastic_concurrent_abstract Abstract

A concurrent algorithm is developed for the serial
\f$\tau\f$-leaping method presented in "Efficient step size selection for the
tau-leaping method" by Yang Cao, Daniel T. Gillespie, and Linda Petzold.
Concurrency is acheived by distributing the reactions among processes.
Concurrent algorithms are developed for both SMP (Symmetric Multi-Processor),
shared-memory architectures and cluster computers.  OpenMP is used for the
former and MPI (Message-Passing Interface) for the latter.
The computational complexity of the serial and the concurrent algorithms
is analyzed.


<!--------------------------------------------------------------------------->
\section stochastic_concurrent_Introduction Introduction

The purpose of this project is to examine the potential usefullness of
concurrent \f$\tau\f$-leaping algorithms.  At this point,
it is not intended to be useful for any other purpose.



<!--------------------------------------------------------------------------->
\section stochastic_concurrent_Previous Previous Work

There has been previous work in utilizing concurrent architectures.  Most
of this work has focused on how to efficiently run a suite of %stochastic
simulations instead of improving the performance of a single simulation.
In StochKit (http://www.engineering.ucsb.edu/~cse/StochKit)
one can use MPI to concurrently collect Monte Carlo ensembles.
This enables one to gather statistics on a collection of simulations.
The approach is to distribute the work of executing the simulations
over the available processors.  This is the simplest way to utilize
a concurrent architecture.  If the number of simulations is significantly
larger than the number of processors, it is also the most effective use
of the concurrent machine.  This is due to its coarse-grained nature and
the small amount of communication required.

StochKit is a general purpose library.  Concurrent programming has also
been applied to specific problems.
In \ref stochastic_langlais2003 "Performance analysis and qualitative results of an efficient parallel %stochastic simulator for a marine host-parasite system"
a hybrid threading/message passing approach is
used with OpenMP and MPI.  As in StochKit, they use MPI to distribute
and gather statistics from an ensemble of simulations.  They also have an
additional layer of concurrency in which they use MPI to perform sensitivity
analysis.  One portion of the algorithm (recruitment of larvae by hosts)
takes most of the computing time in a simulation.  They use OpenMP
(shared-memory) programming to speed up this part of the algorithm.

In summary, the use of MPI to concurrently run a suite of simulations is
well studied.  Threading has been used to improve performance for
some problems.




<!--------------------------------------------------------------------------->
\section stochastic_concurrent_Serial Serial Algorithm


Consider the \f$\tau\f$-leaping method presented in
\ref stochastic_cao2006 "Efficient step size..." .  For
ease of exposition (and implementation), I will cover the new
\f$\tau\f$-selection procedure without the modified
(non-negative) Poisson \f$\tau\f$-leaping.  The concurrent algorithm
that accommodates the non-negative \f$\tau\f$-leaping is analogous.
The table below summarizes the variables in the \f$\tau\f$-leaping method.

<table>
<tr>
<th> Variable
<th> Type
<th> Size
<th> Function
<tr>
<td> \f$\mathbf{a} = \mathbf{a}(\mathbf{X})\f$
<td> Floating point
<td> M
<td> Propensity functions
<tr>
<td> \f$a_0 = \sum_{j=0}^{M-1} a_j(\mathbf{x})\f$
<td> Floating point
<td> 1
<td> Propensity sum
<tr>
<td> v
<td> Integer
<td> \f$\mathcal{O}(M)\f$
<td> State-change vectors
<tr>
<td> \f$\mu_i = \sum_{j=0}^{M-1} \nu_{ij} a_j(\mathbf{x})\f$
<td> Floating point
<td> N
<td> Expected state change
<tr>
<td> \f$\sigma_i^2 = \sum_{j=0}^{M-1} \nu_{ij}^2 a_j(\mathbf{x})\f$
<td> Floating point
<td> N
<td> Expected variance in state change
<tr>
<td> \f$\tau = \min_{i} \left\{ \frac{\max(\epsilon x_i / g_i, 1)}
                                     {|\mu_i(\mathbf{x})|},
                                \frac{\max(\epsilon x_i / g_i, 1)^2}
                                     {|\sigma_i^2(\mathbf{x})|} \right\}\f$
<td> Floating point
<td> 1
<td> Time leap
<tr>
<td> \f$\mathbf{x}\f$
<td> Integer
<td> N
<td> Species populations
</table>


In most scenarios, each reaction involves only a few species.  Thus the
state-change vectors v will be sparse.
There are only \f$\mathcal{O}(M)\f$ nonzero elements.





Below is the pseudo-code for one step of the algorithm.  In the first block
we compute the propensity functions.  Next we compute \f$\mu\f$ and
\f$\sigma^2\f$.  In the third block we determine the \f$\tau\f$ leap.
Finally, we update the state by firing the reactions and advancing the time.
(To obtain simpler formulas, we treat the state change vectors as if they
were dense arrays.)

\verbatim
for j in [0..M):
  a[j] = computePropensityFunction(x, j)

mu = sigmaSquared = 0
for j in [0..M):
  for i in [0..N):
    mu[i] += v[j][i] * a[j]
    sigmaSquared[i] += v[j][i]^2 * a[j]

tau = infinity
for i in [0..N):
  numerator = max(epsilon * x[i] / computeG(x, i), 1)
  temp = min(numerator / abs(mu[i]), numerator^2 / sigmaSquared[i])
  tau = min(tau, temp)

for j in [0..M):
  p = computePoisson(a[i] * tau)
  for i in [0..N):
    x[i] += v[j][i] * p
t += tau \endverbatim


Let \f$T_a\f$ be the cost of an arithmetic operation (integer or floating
point) and \f$T_c\f$ be the cost of a conditional.
Let \f$T_p\f$ be the cost of computing a Poisson random variable.
In the table below we
detail the computational cost in a single step of the \f$\tau\f$-leaping
method on a per variable basis.

<table>
<tr>
<th> Variable
<th> Size
<th> Computational Cost
<tr>
<td> a
<td> M
<td> \f$\mathcal{O}(M) T_a\f$
<tr>
<td> v
<td> \f$\mathcal{O}(M)\f$
<td> 0
<tr>
<td> mu
<td> N
<td> \f$\mathcal{O}(M) T_a\f$
<tr>
<td> sigmaSquared
<td> N
<td> \f$\mathcal{O}(M) T_a\f$
<tr>
<td> tau
<td> 1
<td> \f$N (T_a + T_c)\f$
<tr>
<td> x
<td> N
<td> \f$M T_p + \mathcal{O}(M)T_a\f$
</table>


We see that the computation of \f$\mu\f$, \f$\sigma^2\f$ and x,
and generating the Poisson random variables are
the most expensive parts of the algorithm.  Note that each of these
costs are proportional to the number of reactions.





<!--------------------------------------------------------------------------->
\section stochastic_concurrent_ConcurrentAlgorithm Concurrent Algorithm: Reaction Distributed

We consider a concurrent algorithm for the basic \f$\tau\f$-leaping method on
\f$P\f$ processes.  We will distribute the \f$M\f$ reactions over the processes.
This yields a simple distribution of the data and a simple communication
strategy.  (There is less to be gained in attempting to distribute the
\f$N\f$ species.  They are coupled due to the reactions, so distributing would
be more difficult and the communication costs would be higher.)

In short, we distribute the reaction data and duplicate the species data.
To compute species data, each process computes the contribution from its
set of reactions and then communicates with the other processes to
accumulate the results.  Let
\f$0 = p_0 \leq p_1 \leq \cdots \leq p_{P-1} \leq p_P = M\f$ be a partition of
the \f$M\f$ reactions.  Assume the partition divides the reactions into groups
of size approximately \f$M / P\f$.  If the computational costs associated with
each reaction were roughly equal, this approach would be sufficient to
effectively distribute the computational load.  If this were not the case,
one would consider the different computational costs per reaction in
partitioning them.

For our partition, the process with rank \f$r \in [0..P)\f$ holds
the data for reactions in the range \f$[p_r..p_{r+1})\f$.
Specifically, the propensity functions a and the state change vectors v
are distributed.  All other variables are duplicated.


The pseudo-code for one step of the concurrent algorithm in process \f$r\f$
is shown below.  We use the MPI function
\ref stochastic_mpi "Allreduce()"
to sum and find the minimum across the
processes.  Note that the concurrent algorithm is little different than the
serial one.  The loops over \f$[0..M)\f$ are replaced by loops over
\f$[p_r..p_{r+1})\f$.  We have added an integer array to calculate
the change in the populations.



\verbatim
for j in [p_r..p_{r+1}):
  a[j] = computePropensityFunction(x, j)

mu = sigmaSquared = 0
for j in [p_r..p_{r+1}):
  for i in [0..N):
    mu[i] += v[j][i] * a[j]
    sigmaSquared[i] += v[j][i]^2 * a[j]
communicator.Allreduce(MPI::IN_PLACE, mu, N, MPI::DOUBLE, MPI::SUM)
communicator.Allreduce(MPI::IN_PLACE, sigmaSquared, N, MPI::DOUBLE, MPI::SUM)

tau = infinity
for i in [0..N):
  numerator = max(epsilon * x[i] / computeG(x, i), 1)
  temp = min(numerator / abs(mu[i]), numerator^2 / sigmaSquared[i])
  tau = min(tau, temp)

change = 0
for j in [p_r..p_{r+1}):
  p = computePoisson(a[i] * tau)
  for i in [0..N):
    change[i] += v[j][i] * p
communicator.Allreduce(MPI::IN_PLACE, change, N, MPI::INT, MPI::SUM)
x += change
t += tau \endverbatim


<!--------------------------------------------------------------------------->
\subsection stochastic_concurrent_CommunicationCosts Communication Costs



We employ a simple model for analyzing the cost of communications.
Let \f$T_l\f$ be the communication latency and \f$T_n^{-1}\f$ be the bandwidth
in numbers (integer or floating point) per second.  The cost of
sending or receiving a message of length \f$m\f$ is \f$T_l + m T_n\f$.
The table below shows the computational and
communication costs on a per
variable basis for one step of the \f$\tau\f$-leaping algorithm.




<table>
<tr>
<th> Var.
<th> Dist.
<th> Storage
<th> Computation
<th> Communication
<tr>
<td> a
<td> Yes
<td> M / P
<td> \f$\mathcal{O}(M / P) T_a\f$
<td>
<tr>
<td> v
<td> Yes
<td> \f$\mathcal{O}(M / P)\f$
<td>
<td>
<tr>
<td> mu
<td> No
<td> N
<td> \f$\mathcal{O}(M / P) T_a\f$
<td> \f$(T_l + N T_n) \mathcal{O}(\log P)\f$
<tr>
<td> sigmaSquared
<td> No
<td> N
<td> \f$\mathcal{O}(M / P) T_a\f$
<td> \f$(T_l + N T_n) \mathcal{O}(\log P)\f$
<tr>
<td> tau
<td> No
<td> 1
<td> \f$N (T_a + T_c)\f$
<td>
<tr>
<td> x
<td> No
<td> N
<td> \f$M T_p / P + \mathcal{O}(M / P)T_a + N T_a\f$
<td> \f$(T_l + N T_n) \mathcal{O}(\log P)\f$
</table>



We see that the cost of the most expensive parts of the serial computation
(namely mu, sigmaSquared and x) have all been reduced by a factor
of \f$P\f$.  This is an ideal reduction in the amount of computation each
process
must perform.  However, the concurrent algorithm does have the additional
communication costs.  There are three instances during a time step that
a variable is accumulated over the processes.
(One can reduce this to two by combining mu and sigmaSquared into a single
buffer.)
For most architectures, this is implemented in MPI with the
recursive-doubling algorithm,
hence the \f$\mathcal{O}(\log P)\f$ factor in the complexity.
See \ref stochastic_vandevelde "Concurrent Scientific Computing".








<!--------------------------------------------------------------------------->
\subsection stochastic_concurrent_Overlapping Overlapping Communication and Computation


The pseudo-code above uses the MPI function
Allreduce(), which is a blocking communication.  That is, the function
does not return until the communications are complete.  A common practice
in concurrent computing is using non-blocking communications which allow
a process to continue computing while the communication is taking place.
Unfortunately, this methodology has nothing to offer for the \f$\tau\f$-leaping
method.  mu and sigmaSquared are needed to compute tau.  Thus there
is no computation we can perform while the former are being accumulated
across the processes.  Likewise for accumulating the change in the populations.
Thus there is no opportunity for utilizing non-blocking communications.





<!--------------------------------------------------------------------------->
\subsection stochastic_concurrent_CCC Concurrent Computational Complexity


First assume that <em>N</em> and <em>M</em> are fairly small.  Then the
communication cost is dominated by the latency, i.e. \f$T_l \gg N
T_n\f$.  By comparing the costs of computation and communication, we
expect good scalability when
\f[M T_p / P + N T_a > T_l \log P.\f]
We can separate the terms on the left to obtain two conditions.
\f[M T_p / T_l > P \log P
\quad\mathrm{or}\quad
N T_a / T_l > \log P.\f]
Since the communication latency will be much greater than \f$T_a\f$ or
\f$T_p\f$, the condition is probably not satisfied for any \f$P >1\f$.
Therefore, the concurrent algorithm has nothing to offer for
simulations with small numbers of species and reactions.

A simpler explanation: Message-passing concurrency has nothing to offer
for small problems because the individual steps in tau-leaping are
inexpensive.  If you have a concurrent algorithm that communicates at every
step, then a step had better be fairly expensive.  Otherwise the
communication costs will dominate.


Next consider the case that the numbers of species and reactions are large.
The condition for good scalability is
\f[M T_p / P + N T_a > (T_l + N T_n) \log P.\f]
We assume that the number of species is large enough that the latency cost is
negligible.  Then we split this into two conditions.
\f[M T_p / (N T_n) > P \log P
\quad\mathrm{or}\quad
T_a / T_n > \log P\f]
The second condition won't be true, so our scalability criterion is
\f[M T_p / (N T_n) > P \log P.\f]
The concurrent algorithm has some promise for large problems and more
promise when the number of reactions is much greater than the number of
species.  (One could could have guessed this result without any algebra
by simply recalling that the concurrent algorithm distributes reaction
data and duplicates species data.)


Using the concurrent algorithm offers a modest decrease in the
storage requirement per process since it distributes the reaction data.
This would only be significant for very large problems.



<!--------------------------------------------------------------------------->
\section stochastic_concurrent_Performance Performance

I have tested the concurrent algorithm on Caltech's Shared Heterogeneous
Cluster (SHC).  It consists of 86 Opteron dual-core, dual-processor nodes,
connected via Voltaire's Infiniband and an Extreme Networks BlackDiamond 8810
switch.

The current results are not impressive.  There are two reasons for this:
Firstly, this problem is an example of fine-grained concurrency.
Each step in the algorithm inexpensive.  Thus the cost of message passing
is significant.  A shared-memory implementation would give better results,
because message passing between the processors on a single node is
less efficient than using threads.  Secondly, I have a simple implementation
of tau-leaping.  I don't check for negative populations.  I don't gather
any statistics.  If the algorithm were more sophisticated (and hence costlier)
the relative cost of communication would be lessened.



<!--------------------------------------------------------------------------->
\subsection stochastic_concurrent_DecayingDimerizing Decaying-Dimerizing Problem

Consider the decaying-dimerizing reaction set presented in
\ref stochastic_gillespie2001 "Approximate accelerated %stochastic simulation of chemically reacting systems."
There are three species and four reactions:
\f[
S_1 \rightarrow 0, \quad
S_1 + S_1 \rightarrow S_2, \quad
S_2 \rightarrow S_1 + S_1, \quad
S_2 \rightarrow S_3
\f]
The rate constants are
\f[
c_1 = 1, \quad
c_2 = 0.002 \quad
c_3 = 0.5 \quad
c_4 = 0.04.
\f]
The initial conditions are
\f[
X_1 = 10^5, \quad X_2 = X_3 = 0.
\f]
In the tau-leaping simulations we use \f$\epsilon = 0.03\f$ and run to time
\f$t = 30\f$.  To obtain a problem that is suitable for testing the concurrent
algorithm, we simply duplicate the reactions and species.  That is, the
resulting problem has \f$4 n\f$ reactions and \f$3 n\f$ species.

First we increase the problem size by a factor of 250.

<table>
<tr>
<th> Processors
<td> 1 (serial)
<td> 1
<td> 2
<td> 4
<td> 8
<tr>
<th> Execution time (sec)
<td> 1.54
<td> 1.65
<td> 1.12
<td> 0.92
<td> 1.18
<tr>
<th> Perfect scalability time
<td>
<td> 1.65
<td> 0.83
<td> 0.41
<td> 0.21
<tr>
<th> Reactions per process
<td> 1000
<td> 1000
<td> 500
<td> 250
<td> 125
</table>


Then we increase the problem size by a factor of 2500.

<table>
<tr>
<th> Processors
<td> 1 (serial)
<td> 1
<td> 2
<td> 4
<td> 8
<td> 16
<tr>
<th> Execution time (sec)
<td> 21.73
<td> 22.58
<td> 13.64
<td> 10.21
<td> 8.79
<td> 8.75
<tr>
<th> Perfect scalability time
<td>
<td> 22.58
<td> 11.29
<td> 5.65
<td> 2.82
<td> 1.41
<tr>
<th> Reactions per process
<td> 10000
<td> 10000
<td> 5000
<td> 2500
<td> 1250
<td> 625
</table>



<!--------------------------------------------------------------------------->
\section stochastic_concurrent_SharedMemory Shared-Memory Concurrency Using OpenMP


I have implemented a shared-memory algorithm for the explicit tau-leaping
scheme using OpenMP.  There are two ways to use OpenMP.  The first method
requires minimal changes to the serial source code.  One simply uses
threading to distribute work among the available processors in the
computationally expensive portions of the code.  For example, when one
encounters an expensive loop, one can often use OpenMP directives to
spawn threads and distribute the iterations of the loop.  For some problems
this simple approach works quite well.  For tau-leaping it does not;
a more sophisticated approach is required.  The second method involves
restructuring the algorithm and data structures to utilize concurrency.
Often one needs to duplicate data and communicate among the threads
with shared variables.
(See \ref stochastic_chandra2001 "Parallel Programming in OpenMP" for
a thorough presentation of these issues.)


The difficulty in designing a concurrent algorithm for tau-leaping
comes from the fact that little work is done in a single time step.
It is not worthwhile to use threading on any of the loops in a time
step.  For most problems the cost of spawning threads and distributing
work would dominate the computation.  Instead one must modify the data
structures to effectively use concurrency.  The approach is the same
as that taken for distributed-memory concurrency.  Threads are spawned
at the beginning of the simulation.  The reactions are distributed
among the threads.  The difference is that the threads communicate
through shared variables instead of communicating with messages.


In the shared-memory concurrent algorithm, the reactions are distributed
among the threads.  The species populations array is duplicated.  Each
thread updates its private populations array using its reactions.  (It
would be very costly to have a single population array and use the OpenMP
"critical" directive to update the array.)  After each thread has updated
its population array, the results are collected into the master thread's
array.  The arrays \f$\mu\f$ and \f$\sigma^2\f$ and the time step may
also be computed concurrently.  Since these are much less expensive
than computing the random variables and updating the populations, it is
worthwhile to use concurrency in computing \f$\mu\f$, \f$\sigma^2\f$,
and \f$\tau\f$ only for large problems.  For small problems, one would want
to compute these values on one of the threads and reduce the synchronization
costs.  The implementation supports both modes.  For the large-problem
mode, \f$\mu\f$, \f$\sigma^2\f$, and \f$\tau\f$ are duplicated; each
thread updates its private variables.


<!--------------------------------------------------------------------------->
\subsection stochastic_concurrent_SharedDecayingDimerizing Decaying-Dimerizing Problem

Consider the decaying-dimerizing reaction set presented in
\ref stochastic_gillespie2001 "Approximate accelerated %stochastic simulation of chemically reacting systems."
There are three species and four reactions:
\f[
S_1 \rightarrow 0, \quad
S_1 + S_1 \rightarrow S_2, \quad
S_2 \rightarrow S_1 + S_1, \quad
S_2 \rightarrow S_3
\f]
The rate constants are
\f[
c_1 = 1, \quad
c_2 = 0.002 \quad
c_3 = 0.5 \quad
c_4 = 0.04.
\f]
The initial conditions are
\f[
X_1 = 10^5, \quad X_2 = X_3 = 0.
\f]
In the tau-leaping simulations we use \f$\epsilon = 0.03\f$ and run to time
\f$t = 30\f$.  To obtain a problem that is suitable for testing the concurrent
algorithm, we simply duplicate the reactions and species.  That is, the
resulting problem has \f$4 n\f$ reactions and \f$3 n\f$ species.
I ran tests with values of \f$n\f$ ranging from 5 to 10,000.

The tests were run on an Apple Mac Mini with an 1.66 GHz Intel Core Duo
processor, 512 MB of memory, and a single 2 MB L2 cache that is shared
between the execution cores.  The source code was compiled with GCC 4.2.
The table below shows the execution times in seconds for four scenerios:
- The serial implementation.
- Two instances of the serial implementation run concurrently.
- The concurrent implementation utilizing a single thread.
- The concurrent implementation utilizing two threads.

<table>
<tr>
<th> Reactions
<th> Serial (1 Process)
<th> Serial (2 Processes)
<th> 1 Thread
<th> 2 Threads
<tr><td>20    <td>0.03	<td>0.03   <td>0.04   <td>0.17   <!--<td>0.61-->
<tr><td>40    <td>0.07	<td>0.07   <td>0.07   <td>0.19   <!--<td>0.61-->
<tr><td>100   <td>0.19	<td>0.19   <td>0.19   <td>0.23   <!--<td>0.69-->
<tr><td>200   <td>0.39	<td>0.39   <td>0.4    <td>0.34   <!--<td>0.89-->
<tr><td>400   <td>0.81	<td>0.81   <td>0.83   <td>0.55   <!--<td>1.33-->
<tr><td>1000  <td>2.11	<td>2.12   <td>2.17   <td>1.22   <!--<td>2.67-->
<tr><td>2000  <td>4.35	<td>4.37   <td>4.48   <td>2.39   <!--<td>5.03-->
<tr><td>4000  <td>9	<td>9.21   <td>9.23   <td>4.81   <!--<td>9.95-->
<tr><td>10000 <td>24.05	<td>25.74  <td>24.7   <td>12.97  <!--<td>26.83-->
<tr><td>20000 <td>51.78	<td>56     <td>53.27  <td>28.46  <!--<td>59.14-->
<tr><td>40000 <td>111	<td>119.26 <td>114.45 <td>60.29  <!--<td>125.55-->
</table>
<!--<th> 2 Threads (large)-->
<!--<tr><td>4     <td>0.01	<td>0.01   <td>0.01   <td>0.16--><!--<td>0.57-->


Next we show the execution times of the latter three implementations as a
fraction of the serial execution times.  For the two serial implementations
launched concurrently, we divide by two as this runs a suite of two
simulations.

<table>
<tr>
<th> Reactions
<th> Serial (2 Proc.)
<th> 1 Thread
<th> 2 Threads
<tr><td>20    <td>0.50 <td>1.33 <td> 5.67
<tr><td>40    <td>0.50 <td>1.00 <td> 2.71
<tr><td>100   <td>0.50 <td>1.00 <td> 1.21
<tr><td>200   <td>0.50 <td>1.03 <td> 0.87
<tr><td>400   <td>0.50 <td>1.02 <td> 0.68
<tr><td>1000  <td>0.50 <td>1.03 <td> 0.58
<tr><td>2000  <td>0.50 <td>1.03 <td> 0.55
<tr><td>4000  <td>0.51 <td>1.03 <td> 0.53
<tr><td>10000 <td>0.54 <td>1.03 <td> 0.54
<tr><td>20000 <td>0.54 <td>1.03 <td> 0.55
<tr><td>40000 <td>0.54 <td>1.03 <td> 0.54
</table>
<!--<tr><td>4     <td>0.50 <td>1.00 <td> 16.00-->

\image html decayingDimerizing.jpg "Execution times for the decaying-dimerizing problem."
\image latex decayingDimerizing.pdf "Execution times for the decaying-dimerizing problem." width=0.5\textwidth

First we note that there is little performance difference between the
serial code and the threaded code running with a single thread.
For those developing a tau-leaping implementation, this indicates that
having a serial code is not necessary.  One can simplify the development
by only writing the threaded version.

Next consider the tests in which two serial applications are launched.
Running two serial applications on a dual-core architecture takes longer
than running a single application because the processor speed is not the
only factor that affects performance.  Bus speed, for example, also plays a
role.  The performance results show this effect.  (This effect is not
evident for the small problems because they take such a short time
to execute.  The two processes are not launched exactly simultaneously.)

Finally, consider the tests using two threads.
There are performance benefits in using two threads starting with the
problem that has 200 reactions.  For the larger problems, the two-threaded
versions run in about 54% of the time of the serial version.  This is almost
identical to the (adjusted) time for running two serial applications.
This shows that for large problems, the communication overhead is
negligible.  The concurrent implementation fully utilizes the dual-core
processor.


<!--------------------------------------------------------------------------->
\section stochastic_concurrent_Compilers Compilers for OpenMP and MPI

<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection stochastic_concurrent_CompilersGCC GCC

GCC has support for OpenMP starting with version 4.2.  I have tested the
beta version of 4.2 on a Mac Mini with Intel Core Duo.
I have only encountered minor problems with
this release.  Overall, it has worked very well.

<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection stochastic_concurrent_CompilersPG Portland Group

I have tested version 6.0-4 of the Portland Group compiler on SHC.  It
does not correctly implement barriers.  Some code that uses barriers works
correctly, other code does not.  The compiler will give a nonsensical
compilation error for some barriers.  It says that barriers can ony be
used in parallel blocks.  Wrapping the barrier in braces gets rid of the
compilation error.  However, it seems that the barrier is then ignored.
The current version of the PG compiler is 6.1.  Perhaps they have fixed this
error in that version.  On the positive side, the compiler seems to do a good
job of optimizing the code.  So even though applications do not run correctly,
they run quickly.


<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection stochastic_concurrent_CompilersPathScale PathScale

I have tested version 2.3 of the PathScale compiler on SHC.  It dies
with internal errors when I try to compile any OpenMP application.  I
don't think it supports OpenMP programming with C++.



<!--------------------------------------------------------------------------->
\section stochastic_concurrent_FalseSharing False Sharing

False sharing has a major impact when cores have separate L2 caches.  The
Intel Core Duo has a single L2 cache, so false sharing typically has only a
minor performance impact.  The AMD Opteron chip on SHC has separate L2
caches for its four cores so false sharing can have a major impact on its
performance.



*/

/*!
<!--------------------------------------------------------------------------->
<!--------------------------------------------------------------------------->
\page stochastic_bibliography Bibliography

-# \anchor stochastic_gillespie1977
Daniel T. Gillespie. "Exact Stochastic Simulation of Coupled Chemical
Reactions." Journal of Physical Chemistry, Vol. 81, No. 25, 1977.
-# \anchor stochastic_cao2004
Yang Cao, Hong Li, and Linda Petzold.
"Efficient formulation of the stochastic simulation algorithm for chemically
reacting systems."
Journal of Chemical Physics, Vol. 121, No. 9, 2004.
-# \anchor stochastic_mccollum2005
James M. McCollum, Gregory D. Peterson, Chris D. Cox, Michael L. Simpson, and
Nagiza F. Samatova.
"The sorting direct method for stochastic simulation of biochemical systems
with varying reaction execution behavior."
Computational Biology and Chemistry, Vol. 30, Issue 1, 39-49, Feb. 2006.
-# \anchor stochastic_cao2006
Yang Cao, Daniel T. Gillespie, and Linda R. Petzold.
"Efficient step size selection for the tau-leaping simulation method."
J. Chemical Physics, Vol. 124, No. 4, 2006.
-# \anchor stochastic_gillespie2001
Daniel T. Gillespie.
"Approximate accelerated stochastic simulation of chemically reacting systems."
J. Chemical Physics, Vol. 115, No. 4, 1716-1733, 2001.
-# \anchor stochastic_gibson2000
Michael A. Gibson and Jehoshua Bruck.
"Efficient Exact Stochastic Simulation of Chemical Systems with Many Species
and Many Channels."
Journal of Physical Chemistry A, Vol. 104, No. 9, 1876-1889, 2000.
-# \anchor stochastic_marsaglia2000
George Marsaglia and Wai Wan Tsang.
"The ziggurat method for generating random variables."
Journal of Statistical Software,
Vol. 5, 2000, Issue 8.
http://www.jstatsoft.org/v05/i08/
-# \anchor stochastic_rubin2006
Herman Rubin and Brad Johnson.
"Efficient generation of exponential and normal deviates."
Journal of Statistical Computation and Simulation, Vol. 76, No. 6,
509-518, 2006.
-# \anchor stochastic_Goedecker2001
Stefan Goedecker and Adolfy Hoisie.
"Performance Optimization of Numerically Intensive Codes."
Society for Industrial and Applied Mathematics, Philadelphia, 2001.
-# \anchor stochastic_cormen2001
Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
"Introduction to Algorithms, Second Edition."
The MIT Press, Cambridge, Massachusetts, 2001.
-# \anchor stochastic_chandra2001
Rohit Chandra, Leonardo Dagum, Dave Kohr, Dror Maydan, Jeff McDonald, and
Ramesh Menon.
"Parallel Programming in OpenMP."
Morgan Kaufmann, 2001.
-# \anchor stochastic_mpi
William Gropp, Ewing Lusk, Athony Skjellum.
"Using MPI: Portable Parallel Programming with the Message-Passing Interface."
The MIT Press, 1999.
-# \anchor stochastic_vandevelde
Eric F. Van de Velde.
"Concurrent Scientific Computing."
Springer Verlag, 1994.
-# \anchor stochastic_langlais2003
M. Langais, G. Latu, J. Roman, P. Silan.
"Performance analysis and qualitative results of an efficient parallel
stochastic simulator for a marine host-parasite system."
Concurrency & Computation-Practice & Experience 15 (11-12): 1133-1150 SEP 2003
-# \anchor stochastic_west1979
D. H. D. West, "Updating Mean and Variance Estimates: An Improved Method."
Communications of the ACM, Vol. 22, No. 9, Sep. 1979.

*/


/*
Decaying-Dimerizing on mac mini

Old Data.
<table>
<tr>
<th> Reactions
<th> Serial
<th> 1 Thread
<th> 2 Threads (small)
<th> 2 Threads (large)
<tr><td>4	<td>0.009	<td>0.011	<td>0.199	<td>0.353
<tr><td>20	<td>0.048	<td>0.050	<td>0.223	<td>0.383
<tr><td>40	<td>0.097	<td>0.101	<td>0.241	<td>0.407
<tr><td>200	<td>0.513	<td>0.529	<td>0.469	<td>0.596
<tr><td>400	<td>1.075	<td>1.106	<td>0.776	<td>0.887
<tr><td>2000	<td>5.778	<td>5.939	<td>3.52	<td>3.479
<tr><td>4000	<td>12.009	<td>12.331	<td>7.076	<td>6.866
<tr><td>20000	<td>67.91	<td>69.854	<td>40.886	<td>39.545
<tr><td>40000	<td>146.138	<td>149.754	<td>88.879	<td>84.945
</table>



	Seq		1 thread	2t small	2t large
1	0.017-0.008	0.019-0.008	0.207-0.008	0.362-0.009
5	0.056-0.008	0.059-0.009	0.232-0.009	0.392-0.009
10	0.105-0.008	0.110-0.009	0.250-0.009	0.416-0.009
50	0.524-0.011	0.541-0.012	0.481-0.012	0.608-0.012
100	1.090-0.015	1.122-0.016	0.792-0.016	0.903-0.016
500	5.824-0.046	5.986-0.047	3.568-0.048	3.527-0.048
1000	12.094-0.085	12.416-0.085	7.163-0.087	6.952-0.086
5000	68.338-0.428	70.289-0.435	41.323-0.437	39.983-0.438
10000	146.998-0.86	150.624-0.870	89.756-0.877	85.822-0.877

	Seq		1 thread	2t small	2t large
1	0.009		0.011		0.199		0.353
5	0.048		0.050		0.223		0.383
10	0.097		0.101		0.241		0.407
50	0.513		0.529		0.469		0.596
100	1.075		1.106		0.776		0.887
500	5.778		5.939		3.52		3.479
1000	12.009		12.331		7.076		6.866
5000	67.91		69.854		40.886		39.545
10000	146.138		149.754		88.879		84.945

Decaying-Dimerizing on shc
Small
100
1	2	4
0.74	0.47	0.27

Small
1000
1	2	4
9.63	5.02	3.42

Large
1000
1	2	4
9.19	4.62	2.97

Large
10000
1	2	4
125.56	80.05	42.5


Running on SHC:
qsub -I -l nodes=1

<table>
<tr>
<th> Reactions
<th> Serial
<th> 1 Thread
<th> 2 Threads (small)
<th> 2 Threads (large)

<tr><td>4	<td>2.25	<td>2.75	<td>49.75	<td>88.25
<tr><td>20	<td>2.4		<td>2.5		<td>11.15	<td>19.15
<tr><td>40	<td>2.425	<td>2.525	<td>6.025	<td>10.175
<tr><td>200	<td>2.565	<td>2.645	<td>2.345	<td>2.98
<tr><td>400	<td>2.688	<td>2.765	<td>1.94	<td>2.218
<tr><td>2000	<td>2.889	<td>2.970	<td>1.76	<td>1.740
<tr><td>4000	<td>3.002	<td>3.083	<td>1.769	<td>1.717
<tr><td>20000	<td>3.396	<td>3.493	<td>2.044	<td>1.977
<tr><td>40000	<td>3.653	<td>3.744	<td>2.222	<td>2.124
</table>
*/

#endif

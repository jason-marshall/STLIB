// -*- C++ -*-

/*!
  \file BspTree.h
  \brief Partitioning the elements of a regular grid with a BSP tree.
*/

#if !defined(__concurrent_partition_BspTree_h__)
#define __concurrent_partition_BspTree_h__

#include "stlib/container/MultiArrayRef.h"

#include "stlib/geom/kernel/SemiOpenInterval.h"

namespace stlib
{
namespace concurrent
{

//-----------------------------------------------------------------------------
/*!
  \defgroup concurrent_partition_BspTree Partitioning a Regular Grid with a BSP Tree.

These functions partition the elements elements in a regular grid
(1-D, 2-D, or 3-D).  Each element has an associated cost.  The
algorithm tries to equitably divide the cost while keeping small boundaries
between the partitions.  The size of the boundary between partitions
proportional to the communication costs for adjacent interactions between
the elements.

A binary-space-partition tree is used to divide the grid elements.  The
elements are recursively divided into two groups.  The sub-grid containing
the elements is split along its longest dimension and the elements are
divided to to equitably distribute the cost.  If there are \e N grid
elements and \e P partitions, the computational complexity of the
algorithm in \f$\mathcal{O}(N \log P)\f$.

Because partitioning is fast, I don't see the need for a dynamic algorithm.
Consider a 100 x 100 x 100 grid whose elements costs are drawn from a
uniform random distribution.  Generating partitions of size 10, 100, and 1000
takes 0.08, 0.15, and 0.22 seconds, respectively, on a 1.66 GHz Intel Core
Duo.

Except for very small problems, it is impossible to compute the "best"
partitioning.  There are \f$P^N\f$ ways of partitioning \e N elements into
\c P groups.  If we make the simplifying assumption that each group is of
size \f$N / P\f$, this is reduced to \f$N! / ((N / P)!)^P\f$.
For \e N = 32 and \e P = 4 this is approximately \f$10^{17}\f$.

Below are some examples of partitioning a 16 x 16 grid with unit costs.

\verbatim
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Time to partition 256 elements into 2 groups = 0 seconds.
Elements per partition:
  min = 128, max = 128, mean = 128, sum = 256
Cost per partition:
  min = 128, max = 128, mean = 128, sum = 256, efficiency = 1
Adjacent neighbors for each partition:
  min = 1, max = 1, mean = 1, sum = 2
Send operations for each partition:
  min = 16, max = 16, mean = 16, sum = 32
Ratio of communication to computation:
  min = 0.125, max = 0.125, mean = 0.125 \endverbatim

\verbatim
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2
0 0 0 0 0 1 1 1 1 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Time to partition 256 elements into 3 groups = 0 seconds.
Elements per partition:
  min = 85, max = 86, mean = 85.3333, sum = 256
Cost per partition:
  min = 85, max = 86, mean = 85.3333, sum = 256, efficiency = 0.992248
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 6
Send operations for each partition:
  min = 16, max = 20, mean = 18, sum = 54
Ratio of communication to computation:
  min = 0.188235, max = 0.235294, mean = 0.210944 \endverbatim

\verbatim
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
Time to partition 256 elements into 4 groups = 0 seconds.
Elements per partition:
  min = 64, max = 64, mean = 64, sum = 256
Cost per partition:
  min = 64, max = 64, mean = 64, sum = 256, efficiency = 1
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 8
Send operations for each partition:
  min = 16, max = 16, mean = 16, sum = 64
Ratio of communication to computation:
  min = 0.25, max = 0.25, mean = 0.25 \endverbatim

\verbatim
2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 4
2 2 2 2 2 2 3 3 3 3 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
Time to partition 256 elements into 5 groups = 0 seconds.
Elements per partition:
  min = 51, max = 52, mean = 51.2, sum = 256
Cost per partition:
  min = 51, max = 52, mean = 51.2, sum = 256, efficiency = 0.984615
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 10
Send operations for each partition:
  min = 14, max = 23, mean = 16.6, sum = 83
Ratio of communication to computation:
  min = 0.27451, max = 0.45098, mean = 0.324359 \endverbatim

\verbatim
3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5
3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5
3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5
3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5
3 3 3 3 3 4 4 4 4 4 4 5 5 5 5 5
3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5
3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5
3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
Time to partition 256 elements into 6 groups = 0 seconds.
Elements per partition:
  min = 42, max = 43, mean = 42.6667, sum = 256
Cost per partition:
  min = 42, max = 43, mean = 42.6667, sum = 256, efficiency = 0.992248
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 18
Send operations for each partition:
  min = 13, max = 22, mean = 16, sum = 96
Ratio of communication to computation:
  min = 0.302326, max = 0.52381, mean = 0.375969 \endverbatim

\verbatim
4 4 4 4 4 4 4 6 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
3 3 4 4 4 4 4 4 5 5 5 5 6 6 6 6
3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5
3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5
3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5
3 3 3 3 3 3 3 3 5 5 5 5 5 5 5 5
3 3 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
Time to partition 256 elements into 7 groups = 0 seconds.
Elements per partition:
  min = 36, max = 37, mean = 36.5714, sum = 256
Cost per partition:
  min = 36, max = 37, mean = 36.5714, sum = 256, efficiency = 0.988417
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 14
Send operations for each partition:
  min = 12, max = 21, mean = 15.7143, sum = 110
Ratio of communication to computation:
  min = 0.324324, max = 0.583333, mean = 0.431253 \endverbatim

\verbatim
5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
5 5 5 5 5 5 5 5 7 7 7 7 7 7 7 7
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
4 4 4 4 4 4 4 4 6 6 6 6 6 6 6 6
1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3
1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3
1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3
1 1 1 1 1 1 1 1 3 3 3 3 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
Time to partition 256 elements into 8 groups = 0 seconds.
Elements per partition:
  min = 32, max = 32, mean = 32, sum = 256
Cost per partition:
  min = 32, max = 32, mean = 32, sum = 256, efficiency = 1
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 16
Send operations for each partition:
  min = 12, max = 20, mean = 16, sum = 128
Ratio of communication to computation:
  min = 0.375, max = 0.625, mean = 0.5 \endverbatim

\verbatim
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
4 4 5 5 5 5 5 6 6 6 8 8 8 8 8 7
4 4 4 4 4 4 4 6 6 6 7 7 7 7 7 7
4 4 4 4 4 4 4 6 6 7 7 7 7 7 7 7
4 4 4 4 4 4 4 6 6 7 7 7 7 7 7 7
1 1 4 4 4 4 4 6 6 7 7 7 7 7 7 7
1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 3
1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3
1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
Time to partition 256 elements into 9 groups = 0 seconds.
Elements per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256
Cost per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256, efficiency = 0.980843
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 18
Send operations for each partition:
  min = 11, max = 20, mean = 15.3333, sum = 138
Ratio of communication to computation:
  min = 0.37931, max = 0.714286, mean = 0.54023 \endverbatim

\verbatim
6 6 6 6 6 6 6 7 7 9 9 9 9 9 9 9
6 6 6 6 6 6 6 7 7 9 9 9 9 9 9 9
6 6 6 6 6 6 6 7 7 9 9 9 9 9 9 9
6 6 6 6 6 5 7 7 7 7 8 9 9 9 9 9
5 5 5 5 5 5 7 7 7 7 8 8 8 8 8 8
5 5 5 5 5 5 7 7 7 7 8 8 8 8 8 8
5 5 5 5 5 5 7 7 7 7 8 8 8 8 8 8
5 5 5 5 5 5 7 7 7 7 8 8 8 8 8 8
1 1 1 1 1 1 1 2 2 4 4 4 4 4 4 4
1 1 1 1 1 1 1 2 2 4 4 4 4 4 4 4
1 1 1 1 1 1 1 2 2 4 4 4 4 4 4 4
1 1 1 1 1 0 2 2 2 2 3 4 4 4 4 4
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
Time to partition 256 elements into 10 groups = 0 seconds.
Elements per partition:
  min = 25, max = 26, mean = 25.6, sum = 256
Cost per partition:
  min = 25, max = 26, mean = 25.6, sum = 256, efficiency = 0.984615
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 20
Send operations for each partition:
  min = 9, max = 22, mean = 14.8, sum = 148
Ratio of communication to computation:
  min = 0.346154, max = 0.846154, mean = 0.577846 \endverbatim




Now we show some examples for non-unit costs.  First, the costs are uniformly
distributed random numbers.

\verbatim
6 6 6 6 6 6 7 7 7 7 9 9 9 9 9 9
6 6 6 6 6 6 7 7 7 7 9 9 9 9 9 9
6 6 6 6 6 6 7 7 7 7 9 9 9 9 9 9
6 6 6 6 6 6 7 7 7 7 9 9 9 9 9 9
5 5 6 6 6 6 6 7 7 7 8 8 8 8 8 9
5 5 5 5 5 5 5 7 7 7 8 8 8 8 8 8
5 5 5 5 5 5 5 7 7 7 8 8 8 8 8 8
5 5 5 5 5 5 5 7 7 7 8 8 8 8 8 8
1 1 1 1 1 1 1 2 2 4 4 8 8 8 8 8
1 1 1 1 1 1 1 2 2 4 4 4 4 4 4 4
1 1 1 1 1 1 1 2 2 4 4 4 4 4 4 4
1 1 1 1 0 0 2 2 2 3 4 4 4 4 4 4
0 0 0 0 0 0 2 2 2 3 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 3 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
0 0 0 0 0 0 2 2 2 2 3 3 3 3 3 3
Time to partition 256 elements into 10 groups = 0 seconds.
Elements per partition:
  min = 22, max = 29, mean = 25.6, sum = 256
Cost per partition:
  min = 12.1115, max = 13.0058, mean = 12.4609, sum = 124.609, efficiency = 0.958101
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 20
Send operations for each partition:
  min = 10, max = 19, mean = 14.9, sum = 149
Ratio of communication to computation:
  min = 0.794987, max = 1.54299, mean = 1.19624 \endverbatim

Next, the cost is the first index plus 1.

\verbatim
5 5 5 5 5 5 5 6 6 6 9 9 9 9 9 9
5 5 5 5 5 5 5 6 6 6 9 9 9 9 9 9
5 5 5 5 5 5 5 6 6 6 8 8 9 9 9 9
5 5 5 5 5 5 5 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 5 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 5 6 6 6 8 8 7 7 7 7
5 5 5 5 5 5 5 6 6 6 7 7 7 7 7 7
5 5 5 5 5 5 6 6 6 6 7 7 7 7 7 7
0 0 0 0 0 0 0 1 1 1 4 4 4 4 4 4
0 0 0 0 0 0 0 1 1 1 4 4 4 4 4 4
0 0 0 0 0 0 0 1 1 1 3 3 4 4 4 4
0 0 0 0 0 0 0 1 1 1 3 3 3 3 3 3
0 0 0 0 0 0 0 1 1 1 3 3 3 3 3 3
0 0 0 0 0 0 0 1 1 1 3 3 2 2 2 2
0 0 0 0 0 0 0 1 1 1 2 2 2 2 2 2
0 0 0 0 0 0 1 1 1 1 2 2 2 2 2 2
Time to partition 256 elements into 10 groups = 0 seconds.
Elements per partition:
  min = 16, max = 55, mean = 25.6, sum = 256
Cost per partition:
  min = 208, max = 223, mean = 217.6, sum = 2176, efficiency = 0.975785
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 20
Send operations for each partition:
  min = 8, max = 20, mean = 14.4, sum = 144
Ratio of communication to computation:
  min = 0.0363636, max = 0.0896861, mean = 0.0662375 \endverbatim

Finally, the cost is the sum of the two indices plus 1.

\verbatim
5 5 5 5 5 6 6 6 7 7 7 9 9 9 9 9
5 5 5 5 5 6 6 6 7 7 7 9 9 9 9 9
5 5 5 5 6 6 6 6 7 7 7 9 9 9 9 9
5 5 5 5 6 6 6 6 7 7 7 8 8 8 8 8
5 5 5 5 6 6 6 6 7 7 7 8 8 8 8 8
5 5 5 5 6 6 6 6 7 7 7 8 8 8 8 8
1 1 1 1 1 1 1 1 1 4 4 4 4 4 4 8
1 1 1 1 1 1 1 1 1 4 4 4 4 4 4 4
1 1 1 1 1 1 1 1 1 3 4 4 4 4 4 4
1 1 1 1 1 1 0 0 0 3 3 3 3 3 3 3
0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3
0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3
0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 3
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2
Time to partition 256 elements into 10 groups = 0 seconds.
Elements per partition:
  min = 15, max = 54, mean = 25.6, sum = 256
Cost per partition:
  min = 399, max = 420, mean = 409.6, sum = 4096, efficiency = 0.975238
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 20
Send operations for each partition:
  min = 8, max = 21, mean = 14.4, sum = 144
Ratio of communication to computation:
  min = 0.0190476, max = 0.0514706, mean = 0.0351814 \endverbatim







One can try to predict the optimal partitioning using the shape of the
sub-grid bounding boxes..  This assumes that the costs
are equal and can only be used for small partitions.

\verbatim
4 4 4 4 4 4 7 7 7 7 7 7 8 8 8 8
4 4 4 4 4 4 7 7 7 7 7 8 8 8 8 8
4 4 4 4 4 4 7 7 7 7 7 8 8 8 8 8
4 4 4 4 4 4 7 7 7 7 7 8 8 8 8 8
3 4 4 4 4 4 7 7 7 7 7 8 8 8 8 8
3 3 3 3 3 3 5 5 5 7 7 8 8 8 8 8
3 3 3 3 3 3 5 5 5 5 6 6 6 6 6 6
3 3 3 3 3 5 5 5 5 5 6 6 6 6 6 6
3 3 3 3 3 5 5 5 5 5 6 6 6 6 6 6
3 3 3 3 3 5 5 5 5 5 6 6 6 6 6 6
0 0 0 0 0 5 5 5 5 5 5 6 6 6 6 6
0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2
0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 2
0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2
0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2
Time to partition 256 elements into 9 groups = 0 seconds.
Elements per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256
Cost per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256, efficiency = 0.980843
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 18
Send operations for each partition:
  min = 10, max = 21, mean = 14.2222, sum = 128
Ratio of communication to computation:
  min = 0.344828, max = 0.75, mean = 0.501368 \endverbatim

Compare with the standard algorithm below.

\verbatim
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
5 5 5 5 5 5 6 6 6 6 8 8 8 8 8 8
4 4 5 5 5 5 5 6 6 6 8 8 8 8 8 7
4 4 4 4 4 4 4 6 6 6 7 7 7 7 7 7
4 4 4 4 4 4 4 6 6 7 7 7 7 7 7 7
4 4 4 4 4 4 4 6 6 7 7 7 7 7 7 7
1 1 4 4 4 4 4 6 6 7 7 7 7 7 7 7
1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 3
1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3
1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
0 0 0 0 0 0 0 0 2 2 2 2 3 3 3 3
Time to partition 256 elements into 9 groups = 0 seconds.
Elements per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256
Cost per partition:
  min = 28, max = 29, mean = 28.4444, sum = 256, efficiency = 0.980843
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 18
Send operations for each partition:
  min = 11, max = 20, mean = 15.3333, sum = 138
Ratio of communication to computation:
  min = 0.37931, max = 0.714286, mean = 0.54023 \endverbatim





Next we show a 3-D example.
Consider a 16 x 16 x 16 grid with uniform random costs.  The following tests
partition the grid using partition sizes from 2 to 256.

\verbatim
Time to partition 4096 elements into 2 groups = 0 seconds.
Elements per partition:
  min = 2007, max = 2089, mean = 2048, sum = 4096
Cost per partition:
  min = 1024.12, max = 1024.51, mean = 1024.31, sum = 2048.63, efficiency = 0.999812
Adjacent neighbors for each partition:
  min = 1, max = 1, mean = 1, sum = 2
Send operations for each partition:
  min = 256, max = 256, mean = 256, sum = 512
Ratio of communication to computation:
  min = 0.249877, max = 0.249971, mean = 0.249924 \endverbatim

\verbatim
Time to partition 4096 elements into 4 groups = 0 seconds.
Elements per partition:
  min = 996, max = 1056, mean = 1024, sum = 4096
Cost per partition:
  min = 511.756, max = 512.511, mean = 512.156, sum = 2048.63, efficiency = 0.999309
Adjacent neighbors for each partition:
  min = 2, max = 2, mean = 2, sum = 8
Send operations for each partition:
  min = 253, max = 259, mean = 256, sum = 1024
Ratio of communication to computation:
  min = 0.493648, max = 0.5061, mean = 0.499849 \endverbatim

\verbatim
Time to partition 4096 elements into 8 groups = 0 seconds.
Elements per partition:
  min = 493, max = 528, mean = 512, sum = 4096
Cost per partition:
  min = 255.831, max = 256.385, mean = 256.078, sum = 2048.63, efficiency = 0.998805
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 24
Send operations for each partition:
  min = 186, max = 204, mean = 192.125, sum = 1537
Ratio of communication to computation:
  min = 0.727042, max = 0.797108, mean = 0.75026 \endverbatim

\verbatim
Time to partition 4096 elements into 16 groups = 0 seconds.
Elements per partition:
  min = 243, max = 269, mean = 256, sum = 4096
Cost per partition:
  min = 127.81, max = 128.315, mean = 128.039, sum = 2048.63, efficiency = 0.997848
Adjacent neighbors for each partition:
  min = 4, max = 4, mean = 4, sum = 64
Send operations for each partition:
  min = 118, max = 202, mean = 159.75, sum = 2556
Ratio of communication to computation:
  min = 0.922425, max = 1.57819, mean = 1.24753 \endverbatim

\verbatim
Time to partition 4096 elements into 32 groups = 0 seconds.
Elements per partition:
  min = 115, max = 139, mean = 128, sum = 4096
Cost per partition:
  min = 63.5288, max = 64.4265, mean = 64.0196, sum = 2048.63, efficiency = 0.993683
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 96
Send operations for each partition:
  min = 74, max = 158, mean = 112.406, sum = 3597
Ratio of communication to computation:
  min = 1.16332, max = 2.46842, mean = 1.75556 \endverbatim

\verbatim
Time to partition 4096 elements into 64 groups = 0 seconds.
Elements per partition:
  min = 55, max = 73, mean = 64, sum = 4096
Cost per partition:
  min = 31.5127, max = 32.4949, mean = 32.0098, sum = 2048.63, efficiency = 0.98507
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 192
Send operations for each partition:
  min = 44, max = 109, mean = 72.4062, sum = 4634
Ratio of communication to computation:
  min = 1.38272, max = 3.43516, mean = 2.26184 \endverbatim

\verbatim
Time to partition 4096 elements into 128 groups = 0 seconds.
Elements per partition:
  min = 25, max = 39, mean = 32, sum = 4096
Cost per partition:
  min = 15.3814, max = 16.5314, mean = 16.0049, sum = 2048.63, efficiency = 0.968152
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 384
Send operations for each partition:
  min = 25, max = 74, mean = 51.9688, sum = 6652
Ratio of communication to computation:
  min = 1.55176, max = 4.65012, mean = 3.2483 \endverbatim

\verbatim
Time to partition 4096 elements into 256 groups = 0 seconds.
Elements per partition:
  min = 10, max = 24, mean = 16, sum = 4096
Cost per partition:
  min = 7.36748, max = 8.62067, mean = 8.00244, sum = 2048.63, efficiency = 0.928286
Adjacent neighbors for each partition:
  min = 3, max = 3, mean = 3, sum = 768
Send operations for each partition:
  min = 16, max = 57, mean = 34.3984, sum = 8806
Ratio of communication to computation:
  min = 2.01732, max = 7.06333, mean = 4.30354 \endverbatim


We see that the efficiency stays high, but that the communication costs
climb with the partition size.  This information is summarized in the table
below.  From this one can assess the potential benefits of using mixed-mode
(message passing and shared memory) programming versus message passing
on a cluster of multi-core nodes.

<table>
<tr> <th> Partition Size <th> Efficiency <th> Total Communication <th> Mean Ratio of Communication to Computation
<tr> <td> 2 <td> 0.9998 <td> 512 <td> 0.25
<tr> <td> 4 <td> 0.9993 <td> 1024 <td> 0.50
<tr> <td> 8 <td> 0.9988 <td> 1537 <td> 0.75
<tr> <td> 16 <td> 0.9978 <td> 2556 <td> 1.25
<tr> <td> 32 <td> 0.9936 <td> 3597 <td> 1.76
<tr> <td> 64 <td> 0.9850 <td> 4634 <td> 2.26
<tr> <td> 128 <td> 0.9681 <td> 6652 <td> 3.25
<tr> <td> 256 <td> 0.9282 <td> 8806 <td> 4.30
</table>
*/
//@{

// CONTINUE: Switch to ConstRef once MultiArray has been fixed.

//! Partition a 1-D cost array with a binary-space-partion tree.
template<typename _T>
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 1>& costs,
 container::MultiArrayRef<std::size_t, 1>* identifiers,
 std::size_t numberOfPartitions);

//! Partition a 2-D cost array with a binary-space-partion tree.
template<typename _T>
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 2>& costs,
 container::MultiArrayRef<std::size_t, 2>* identifiers,
 std::size_t numberOfPartitions,
 std::size_t predictionThreshhold = 0);

//! Partition a 3-D cost array with a binary-space-partion tree.
template<typename _T>
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 3>& costs,
 container::MultiArrayRef<std::size_t, 3>* identifiers,
 std::size_t numberOfPartitions);

//@}

} // namespace concurrent
}

#define __partition_BspTree_ipp__
#include "stlib/concurrent/partition/BspTree.ipp"
#undef __partition_BspTree_ipp__

#endif

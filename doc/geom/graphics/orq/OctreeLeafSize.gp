# OctreeLeafSize.gp

#
# Chair problem
#

set size 0.4,0.4
set logscale x
set nokey
set title 'Octree, Leaf Size, Chair'
set xtics 2,2,64
set nomxtics
set ytics 0,2,10
set nomytics
set xlabel 'Leaf Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'OctreeLeafSizeChairTime.eps'
plot [:] [0:] 'OctreeLeafSizeChairTime.dat' with linespoints

set ytics 0,2,10
set ylabel 'Memory Usage (Mb)'
set output 'OctreeLeafSizeChairMemory.eps'
plot [:] [0:] 'OctreeLeafSizeChairMemory.dat' with linespoints

#
# Random points problem
#

set title 'Octree, Leaf Size, Random Points'
set ytics 0,5,30
set ylabel 'Execution Time (s)'
set output 'OctreeLeafSizeRandomTime.eps'
plot [:] [0:] 'OctreeLeafSizeRandomTime.dat' with linespoints

set ytics 0,2,8
set ylabel 'Memory Usage (Mb)'
set output 'OctreeLeafSizeRandomMemory.eps'
plot [:] [0:] 'OctreeLeafSizeRandomMemory.dat' with linespoints

set logscale y
set xtics 1,10,1000
set ytics 4,2,64
set title 'Octree, Best Leaf Size, Random'
set xlabel 'Records per Query'
set ylabel 'Best Leaf Size'
set output 'OctreeLeafSizeRandomBestSize.eps'
plot [1.78504/1.5:2549.25*1.5] [4/1.2:64*1.2] 'OctreeLeafSizeRandomBest.dat' with points

set ytics 0.02,4,1.28
set title 'Octree, Best Size Ratio, Random'
set xlabel 'Records per Query'
set ylabel 'Leaf Size / Query Size'
set output 'OctreeLeafSizeRandomBestRatio.eps'
f(x,y) = y / x
plot [1.78504/1.5:2549.25*1.5] [64/2549.25/1.5:4/1.78504*1.5] \
  'OctreeLeafSizeRandomBest.dat' using 1:($2/$1) with points

#1.78504   4.
#2549.25   64.

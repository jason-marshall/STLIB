# KDTreeLeafSize.gp

#
# Chair problem
#

set size 0.4,0.4
set logscale x
set nokey
set title 'Kd-tree, Leaf Size, Chair'
set xtics 2,2,64
set nomxtics
set ytics 0,10,35
set nomytics
set xlabel 'Leaf Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'KDTreeLeafSizeChairTime.eps'
plot [:] [0:] 'KDTreeLeafSizeChairTime.dat' with linespoints

set ytics 0,1,4
set ylabel 'Memory Usage (Mb)'
set output 'KDTreeLeafSizeChairMemory.eps'
plot [:] [0:] 'KDTreeLeafSizeChairMemory.dat' with linespoints

#
# Random points problem
#

set title 'Kd-tree, Leaf Size, Random Points'
set ytics 0,4,24
set ylabel 'Execution Time (s)'
set output 'KDTreeLeafSizeRandomTime.eps'
plot [:] [0:] 'KDTreeLeafSizeRandomTime.dat' with linespoints

set ytics 0,1,3
set ylabel 'Memory Usage (Mb)'
set output 'KDTreeLeafSizeRandomMemory.eps'
plot [:] [0:] 'KDTreeLeafSizeRandomMemory.dat' with linespoints

#
# Random points problem, best leaf size
#

set logscale y
set xtics 1,10,1000
set ytics 4,2,32
set title 'Kd-tree, Best Leaf Size, Random'
set xlabel 'Records per Query'
set ylabel 'Best Leaf Size'
set output 'KDTreeLeafSizeRandomBestSize.eps'
plot [1.78504/1.5:2549.25*1.5] [4/1.2:32*1.2] 'KDTreeLeafSizeRandomBest.dat' with points

set ytics 0.02,4,1.28
set title 'Kd-tree, Best Size Ratio, Random'
set xlabel 'Records per Query'
set ylabel 'Leaf Size / Query Size'
set output 'KDTreeLeafSizeRandomBestRatio.eps'
f(x,y) = y / x
plot [1.78504/1.5:2549.25*1.5] [8/701.709/1.5:4/1.78504*1.5] \
  'KDTreeLeafSizeRandomBest.dat' using 1:($2/$1) with points

#1.78504   4.
#701.709   8.
#2549.25   32.

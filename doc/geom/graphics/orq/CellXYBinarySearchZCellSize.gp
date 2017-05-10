# CellXYBinarySearchZCellSize.gp

#
# Chair problem
#

set size 0.4,0.4
#set logscale x
set key left top
set title 'Cells Binary Search, Cell Size, Chair'
set xtics 0.5,0.5,2
set nomxtics
set ytics 0,4,20
set nomytics
set xlabel 'Cell Size / Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'CellXYBinarySearchZCellSizeChairTime.eps'
plot [0:] [0:] 'CellXYBinarySearchZCellSizeChairTime.dat' \
  using 1:2 title "Search" with linespoints, \
  'CellXYBinarySearchZCellSizeChairTime.dat' \
  using 1:3 title "Sparse" with linespoints

set key right top 
set ytics 0,0.5,3
set ylabel 'Memory Usage (Mb)'
set output 'CellXYBinarySearchZCellSizeChairMemory.eps'
plot [0:] [0:] 'CellXYBinarySearchZCellSizeChairMemory.dat' \
  using 1:2 title "Search" with linespoints, \
  'CellXYBinarySearchZCellSizeChairMemory.dat' \
  using 1:3 title "Sparse" with linespoints

#
# Random points
#

set key left top 
set title 'Cells Binary Search, Cell Size, Random'
set ytics 0,10,100
set ylabel 'Execution Time (s)'
set output 'CellXYBinarySearchZCellSizeRandomTime.eps'
plot [0:] [0:] 'CellXYBinarySearchZCellSizeRandomTime.dat' \
  using 1:2 title "Search" with linespoints, \
  'CellXYBinarySearchZCellSizeRandomTime.dat' \
  using 1:3 title "Sparse" with linespoints

set key right top 
set ytics 0,0.5,3
set ylabel 'Memory Usage (Mb)'
set output 'CellXYBinarySearchZCellSizeRandomMemory.eps'
plot [0:] [0:] 'CellXYBinarySearchZCellSizeRandomMemory.dat' \
  using 1:2 title "Search" with linespoints, \
  'CellXYBinarySearchZCellSizeRandomMemory.dat' \
  using 1:3 title "Sparse" with linespoints

#
# Best Cell Size
#

set nologscale y
set logscale x
set nokey
set xtics 0.01,4,0.16
set ytics 0.0425,0.0025,0.05
set title 'Cells B. S., Best Cell Size, Random'
set xlabel 'Query Size'
set ylabel 'Best Cell Size'
set output 'CellXYBinarySearchZCellSizeRandomBestSize.eps'
plot [0.01/1.2:0.16*1.2] [0.04:0.0525] \
  'CellXYBinarySearchZCellSizeRandomBestSize.dat' with points

set logscale y
set ytics 0.25,2,4
set title 'Cells B. S., Best Size Ratio, Random'
set ylabel 'Cell Size / Query Size'
set output 'CellXYBinarySearchZCellSizeRandomBestSizeRatio.eps'
plot [0.01/1.2:0.16*1.2] [0.045/0.16/1.2:0.045/0.01*1.2] \
  'CellXYBinarySearchZCellSizeRandomBestSize.dat' using 1:($2/$1) with points

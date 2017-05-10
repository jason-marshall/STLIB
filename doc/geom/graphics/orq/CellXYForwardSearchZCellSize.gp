# CellXYForwardSearchZCellSize.gp

#
# Chair problem
#

set size 0.4,0.4
set key left top
set title 'Cells Forward Search, Cell Size, Chair'
set xtics 0.5,0.5,2
set nomxtics
#set ytics 4,2,16
set ytics 0,2,20
set nomytics
set xlabel 'Cell Size / Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'CellXYForwardSearchZCellSizeChairTime.eps'
plot [0:] [0:] 'CellXYForwardSearchZCellSizeChairTime.dat' \
  using 1:2 title "Forward" with linespoints, \
  'CellXYForwardSearchZCellSizeChairTime.dat' \
  using 1:3 title "Binary" with linespoints

set title 'Cells Forward Search, Cell Size, Chair' -1
set key right top 
set ytics 0,0.5,3
set ylabel 'Memory Usage (Mb)'
set output 'CellXYForwardSearchZCellSizeChairMemory.eps'
plot [0:] [0:] 'CellXYForwardSearchZCellSizeChairMemory.dat' \
  using 1:2 title "Forward" with linespoints, \
  'CellXYForwardSearchZCellSizeChairMemory.dat' \
  using 1:3 title "Binary" with linespoints

#
# Random points
#

set key left top
set title 'Cells Forward Search, Cell Size, Random'
set ytics 0,5,100
set ylabel 'Execution Time (s)'
set output 'CellXYForwardSearchZCellSizeRandomTime.eps'
plot [0:] [0:] 'CellXYForwardSearchZCellSizeRandomTime.dat' \
  using 1:2 title "Forward" with linespoints, \
  'CellXYForwardSearchZCellSizeRandomTime.dat' \
  using 1:3 title "Binary" with linespoints

set key right bottom
set ytics 0,0.5,3
set ylabel 'Memory Usage (Mb)'
set output 'CellXYForwardSearchZCellSizeRandomMemory.eps'
plot [0:] [0:] 'CellXYForwardSearchZCellSizeRandomMemory.dat' \
  using 1:2 title "Forward" with linespoints, \
  'CellXYForwardSearchZCellSizeRandomMemory.dat' \
  using 1:3 title "Binary" with linespoints

#0.1    1.04011    0.560108
#2.     0.800708   0.400508

#
# Best Cell Size
#

set nologscale y
set logscale x
set nokey
set xtics 0.01,4,0.16
set ytics 0.025,0.005,0.035
set title 'Cells Forward S., Best Cell Size, Random' -2
set xlabel 'Query Size'
set ylabel 'Best Cell Size'
set output 'CellXYForwardSearchZCellSizeRandomBestSize.eps'
plot [0.01/1.1:0.16*1.1] [0.025:0.0375] \
  'CellXYForwardSearchZCellSizeRandomBestSize.dat' with points

set logscale y
set ytics 0.25,2,4
set title 'Cells Forward S., Best Size Ratio, Random' -2
set ylabel 'Cell Size / Query Size'
set output 'CellXYForwardSearchZCellSizeRandomBestSizeRatio.eps'
plot [0.01/1.1:0.16*1.1] [0.03/0.16/1.1:0.0275/0.01*1.1] \
  'CellXYForwardSearchZCellSizeRandomBestSize.dat' using 1:($2/$1) with points


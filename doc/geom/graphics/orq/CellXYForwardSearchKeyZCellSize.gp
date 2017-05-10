# CellXYForwardSearchKeyZCellSize.gp

#
# Chair problem
#

set size 0.4,0.4
set key graph 0.5,0.9
set title 'Cells F. S. Store Keys, Cell Size, Chair'
set xtics 0.5,0.5,2
set nomxtics
set ytics 0,2,20
set nomytics
set xlabel 'Cell Size / Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'CellXYForwardSearchKeyZCellSizeChairTime.eps'
plot [0:] [0:] 'CellXYForwardSearchKeyZCellSizeChairTime.dat' \
  using 1:2 title "Store Keys" with linespoints, \
  'CellXYForwardSearchKeyZCellSizeChairTime.dat' \
  using 1:3 title "Look Up Keys" with linespoints

set key graph 0.8,0.5
set ytics 0,1,10
set ylabel 'Memory Usage (Mb)'
set output 'CellXYForwardSearchKeyZCellSizeChairMemory.eps'
plot [0:] [0:] 'CellXYForwardSearchKeyZCellSizeChairMemory.dat' \
  using 1:2 title "Store Keys" with linespoints, \
  'CellXYForwardSearchKeyZCellSizeChairMemory.dat' \
  using 1:3 title "Look Up Keys" with linespoints
#2.         3.72553   0.932364

#
# Random points
#

set key left top
set title 'Cells F. S. Store Keys, Cell Size, Random' -1
set ytics 0,4,20
set ylabel 'Execution Time (s)'
set output 'CellXYForwardSearchKeyZCellSizeRandomTime.eps'
plot [0:] [0:] 'CellXYForwardSearchKeyZCellSizeRandomTime.dat' \
  using 1:2 title "Store Keys" with linespoints, \
  'CellXYForwardSearchKeyZCellSizeRandomTime.dat' \
  using 1:3 title "Look Up Keys" with linespoints
#0.5    2.28096   4.43853

set key graph 0.8,0.5
set ytics 0,1,5
set ylabel 'Memory Usage (Mb)'
set output 'CellXYForwardSearchKeyZCellSizeRandomMemory.eps'
plot [0:] [0:] 'CellXYForwardSearchKeyZCellSizeRandomMemory.dat' \
  using 1:2 title "Store Keys" with linespoints, \
  'CellXYForwardSearchKeyZCellSizeRandomMemory.dat' \
  using 1:3 title "Look Up Keys" with linespoints
#2.     3.20161   0.800708


quit

# Not used

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
set output 'CellXYForwardSearchKeyZCellSizeRandomBestSize.eps'
plot [0.01/1.1:0.16*1.1] [0.025:0.0375] \
  'CellXYForwardSearchKeyZCellSizeRandomBestSize.dat' with points

set logscale y
set ytics 0.25,2,4
set title 'Cells Forward S., Best Size Ratio, Random' -2
set ylabel 'Cell Size / Query Size'
set output 'CellXYForwardSearchKeyZCellSizeRandomBestSizeRatio.eps'
plot [0.01/1.1:0.16*1.1] [0.03/0.16/1.1:0.0275/0.01*1.1] \
  'CellXYForwardSearchKeyZCellSizeRandomBestSize.dat' using 1:($2/$1) with points


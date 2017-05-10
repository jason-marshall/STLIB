# SparseCellArrayCellSize.gp

#
# Chair problem
#

set size 0.4,0.4
set key left top
set title 'Sparse Cell Array, Cell Size, Chair'
set xtics 0.5,0.5,2
set nomxtics
set ytics 0,4,16
set nomytics
set xlabel 'Cell Size / Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'SparseCellArrayCellSizeChairTime.eps'
plot [0:] [0:] 'SparseCellArrayCellSizeChairTime.dat' \
  using 1:2 title "Sparse" with linespoints, \
  'SparseCellArrayCellSizeChairTime.dat' \
  using 1:3 title "Dense" with linespoints

set key right top 
set ytics 0,1,4
set ylabel 'Memory Usage (Mb)'
set output 'SparseCellArrayCellSizeChairMemory.eps'
plot [0:] [0:4] 'SparseCellArrayCellSizeChairMemory.dat' \
  using 1:2 title "Sparse" with linespoints, \
  'SparseCellArrayCellSizeChairMemory.dat' \
  using 1:3 title "Dense" with linespoints

#
# Random points
#

set key left top 
set title 'Sparse Cell Array, Cell Size, Random'
set ytics 0,10,100
set ylabel 'Execution Time (s)'
set output 'SparseCellArrayCellSizeRandomTime.eps'
plot [0:] [0:] 'SparseCellArrayCellSizeRandomTime.dat' \
  using 1:2 title "Sparse" with linespoints, \
  'SparseCellArrayCellSizeRandomTime.dat' \
  using 1:3 title "Dense" with linespoints

set key right top 
set ytics 0,1,4
set ylabel 'Memory Usage (Mb)'
set output 'SparseCellArrayCellSizeRandomMemory.eps'
plot [0:] [0:4] 'SparseCellArrayCellSizeRandomMemory.dat' \
  using 1:2 title "Sparse" with linespoints, \
  'SparseCellArrayCellSizeRandomMemory.dat' \
  using 1:3 title "Dense" with linespoints

#
# Best Cell Size
#

set logscale xy
set nokey
set xtics 0.01,4,0.16
set ytics 0.02,1.5,0.045
set title 'Sparse Cells, Best Cell Size, Random'
set xlabel 'Query Size'
set ylabel 'Best Cell Size'
set output 'SparseCellArrayCellSizeRandomBestSize.eps'
plot [0.01/1.2:0.16*1.2] [0.02/1.1:0.045*1.1] \
  'SparseCellArrayCellSizeRandomBestSize.dat' with points

set ytics 0.25,2,2
set title 'Sparse Cells, Best Size Ratio, Random'
set ylabel 'Cell Size / Query Size'
set output 'SparseCellArrayCellSizeRandomBestSizeRatio.eps'
plot [0.01/1.2:0.16*1.2] [0.045/0.16/1.2:0.02/0.01*1.2] \
  'SparseCellArrayCellSizeRandomBestSize.dat' using 1:($2/$1) with points

set xtics 1,10,1000
set ytics 0.004,4,0.256
set title 'Sparse Cells, Record Ratio, Random'
set xlabel 'Query Records'
set ylabel 'Cell Records / Query Records'
set output 'SparseCellArrayCellSizeRandomBestRecordRatio.eps'
plot [1.78504/1.5:2549.25*1.5] [9.1125/2549.25/1.5:0.8/1.78504*1.5] \
  'SparseCellArrayCellSizeRandomBestRecord.dat' using 1:($2/$1) with points

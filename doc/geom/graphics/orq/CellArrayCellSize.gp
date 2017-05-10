# CellArrayCellSize.gp

#
# Chair problem
#

set size 0.4,0.4
set nokey
set title 'Cell Array, Cell Size, Chair'
set xtics 0.5,0.5,2
set nomxtics
set ytics 0,4,20
set nomytics
set xlabel 'Cell Size / Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'CellArrayCellSizeChairTime.eps'
plot [0:] [0:] 'CellArrayCellSizeChairTime.dat' with linespoints

set logscale y
set ytics 1,4,64
set ylabel 'Memory Usage (Mb)'
set output 'CellArrayCellSizeChairMemory.eps'
plot [0:] [0.4:] 'CellArrayCellSizeChairMemory.dat' with linespoints


#
# Random points
#

set nologscale y
set title 'Cell Array, Cell Size, Random Points'
set ytics 0,10,100
set ylabel 'Execution Time (s)'
set output 'CellArrayCellSizeRandomTime.eps'
plot [0:] [0:] 'CellArrayCellSizeRandomTime.dat' with linespoints

set logscale y
set ytics 0.5,2,16
set ylabel 'Memory Usage (Mb)'
set output 'CellArrayCellSizeRandomMemory.eps'
plot [0:] [0.35:] 'CellArrayCellSizeRandomMemory.dat' with linespoints

#
# Best Cell Size
#

set logscale xy
set xtics 0.01,4,0.16
set ytics 0.02,1.5,0.03
set title 'Cell Array, Best Cell Size, Random'
set xlabel 'Query Size'
set ylabel 'Best Cell Size'
set output 'CellArrayCellSizeRandomBestSize.eps'
plot [0.01/1.2:0.16*1.2] [0.0175/1.1:0.0375*1.1] \
  'CellArrayCellSizeRandomBestSize.dat' with points

set ytics 0.25,2,2
set title 'Cell Array, Best Size Ratio, Random'
set ylabel 'Cell Size / Query Size'
set output 'CellArrayCellSizeRandomBestSizeRatio.eps'
plot [0.01/1.2:0.16*1.2] [0.0375/0.16/1.2:0.0175/0.01*1.2] \
  'CellArrayCellSizeRandomBestSize.dat' using 1:($2/$1) with points

set xtics 1,10,1000
set ytics 0.002,4,0.128
set title 'Cell Array, Record Ratio, Random'
set xlabel 'Query Records'
set ylabel 'Cell Records / Query Records'
set output 'CellArrayCellSizeRandomBestRecordRatio.eps'
plot [1.78504/1.5:2549.25*1.5] [5.27344/2549.25/1.5:0.8/2.56454*1.5] \
  'CellArrayCellSizeRandomBestRecord.dat' using 1:($2/$1) with points

#0.01        0.0175
#0.16        0.0375

#1.78504   0.535938
#2.56454   0.8
#2549.25   5.27344

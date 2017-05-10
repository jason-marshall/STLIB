# CompareFileSizeChair.gp

set size 0.7,0.7
set key outside
set title 'Comparison for Different File Sizes, Chair Problems'
set xlabel 'Number of Reported Records'
set ylabel 'Time per Reported Record (microsec)'
set terminal postscript eps color
set logscale xy
set xtics 10000,10,10000000
set nomxtics
set ytics 0.25,2,256
set nomytics
set output 'CompareFileSizeChairTime.eps'
plot [65412./1.1:47768216*1.1] [0.18:32]\
  'CompareFileSizeChairTime.dat'\
  using 2:3 title "seq. scan" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:4 title "projection" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:5 title "pt-in-box" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:6 title "kd-tree" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:7 title "kd-tree d." with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:8 title "octree" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:9 title "cell" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:10 title "sparse cell" with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:11 title "cell b. s." with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:12 title "cell f. s." with linespoints,\
  'CompareFileSizeChairTime.dat'\
  using 2:13 title "cell f. s. k." with linespoints

set xlabel 'Number of Records in File'
set ylabel 'Memory per Record (bytes)'
set nologscale y
set ytics 0,8,40
set output 'CompareFileSizeChairMemory.eps'
plot [1782./1.1:1864200*1.1] [0:41]\
  'CompareFileSizeChairMemory.dat'\
  using 1:2 title "seq. scan" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:3 title "projection" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:4 title "pt-in-box" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:5 title "kd-tree" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:6 title "kd-tree d." with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:7 title "octree" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:8 title "cell" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:9 title "sparse cell" with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:10 title "cell b. s." with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:11 title "cell f. s." with linespoints,\
  'CompareFileSizeChairMemory.dat'\
  using 1:12 title "cell f. s. k." with linespoints

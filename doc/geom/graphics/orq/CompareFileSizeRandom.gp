# CompareFileSizeRandom.gp

set size 0.7,0.7
set key outside
set title 'Comparison for Different File Sizes, Random Points'
set xlabel 'Number of Reported Records'
set ylabel 'Time per Reported Record (microsec)'
set terminal postscript eps color
set logscale xy
set xtics 100,10,10000000
set nomxtics
set ytics 0.25,2,256
set nomytics
set output 'CompareFileSizeRandomTime.eps'
plot [886/1.15:10842624*1.15] [0.5:9]\
  'CompareFileSizeRandomTime.dat'\
  using 2:3 title "seq. scan" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:4 title "projection" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:5 title "pt-in-box" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:6 title "kd-tree" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:7 title "kd-tree d." with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:8 title "octree" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:9 title "cell" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:10 title "sparse cell" with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:11 title "cell b. s." with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:12 title "cell f. s." with linespoints,\
  'CompareFileSizeRandomTime.dat'\
  using 2:13 title "cell f. s. k." with linespoints

set xlabel 'Number of Records in File'
set ylabel 'Memory per Record (bytes)'
set nologscale y
set ytics 0,8,40
set output 'CompareFileSizeRandomMemory.eps'
plot [100/1.15:1000000*1.15] [0:48]\
  'CompareFileSizeRandomMemory.dat'\
  using 1:2 title "seq. scan" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:3 title "projection" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:4 title "pt-in-box" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:5 title "kd-tree" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:6 title "kd-tree d." with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:7 title "octree" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:8 title "cell" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:9 title "sparse cell" with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:10 title "cell b. s." with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:11 title "cell f. s." with linespoints,\
  'CompareFileSizeRandomMemory.dat'\
  using 1:12 title "cell f. s. k." with linespoints

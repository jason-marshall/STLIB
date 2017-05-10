# TreeFileSizeChair.gp

set size 0.5,0.5
set key graph 0.8,0.5
set title 'Tree Methods, File Size, Chair'
set xlabel 'Number of Records in File'
set ylabel 'Time per Reported Record (microsec)'
set terminal postscript eps color
set logscale xy
set xtics 1000,10,1000000
set nomxtics
set ytics 0.5,2,4
set nomytics
set output 'TreeFileSizeChairTime.eps'
plot [1782./1.1:1864200*1.1]\
  'TreeFileSizeChairTime.dat'\
  using 1:3 title "kd-tree" with linespoints,\
  'TreeFileSizeChairTime.dat'\
  using 1:4 title "kd-tree w/ domain checking" with linespoints,\
  'TreeFileSizeChairTime.dat'\
  using 1:5 title "octree" with linespoints

set ylabel 'Memory per Record (bytes)'
set ytics 10,2,40
set output 'TreeFileSizeChairMemory.eps'
plot [1782./1.1:1864200*1.1]\
  'TreeQuerySizeRandomCube.dat'\
  using 2:($3/$2*10e6) title "kd-tree" with linespoints,\
  'TreeQuerySizeRandomCube.dat'\
  using 2:($4/$2*10e6) title "kd-tree w/ domain checking" with linespoints,\
  'TreeQuerySizeRandomCube.dat'\
  using 2:($5/$2*10e6) title "octree" with linespoints
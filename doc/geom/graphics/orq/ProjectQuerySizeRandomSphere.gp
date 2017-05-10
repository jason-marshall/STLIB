# ProjectQuerySizeRandomSphere.gp

set size 0.5,0.5
set key right bottom
set title 'Projection Methods, Random on Sphere'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set logscale xy
set xtics 0.01,10,1
set nomxtics
#set ytics 0.00025,4,1
set ytics 0.001,10,1
set nomytics
set output 'ProjectQuerySizeRandomSphereTotal.eps'
plot [0.00138107/1.1:2*1.1]\
  'ProjectQuerySizeRandomSphere.dat'\
  using 1:3 title "Projection" with linespoints,\
  'ProjectQuerySizeRandomSphere.dat'\
  using 1:4 title "Point-In-Box" with linespoints,\
  'ProjectQuerySizeRandomSphere.dat'\
  using 1:5 title "Sequential Scan" with linespoints

# CellQuerySizeRandomSphere.gp

set size 0.7,0.7
set key right bottom
set title 'Cell Methods, Random Points on a Sphere'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set logscale xy
set xtics 0.0025,4,2
set nomxtics
set ytics 0.00001,10,0.1
set nomytics
set output 'CellQuerySizeRandomSphereTotal.eps'
plot [0.00138107/1.1:2*1.1]\
  'CellQuerySizeRandomSphere.dat'\
  using 1:3 title "Cell Array" with linespoints,\
  'CellQuerySizeRandomSphere.dat'\
  using 1:4 title "Sparse Cell Array" with linespoints,\
  'CellQuerySizeRandomSphere.dat'\
  using 1:5 title "Cells w/ Binary Searching" with linespoints,\
  'CellQuerySizeRandomSphere.dat'\
  using 1:6 title "Sequential Scan" with linespoints

set key right top
set xlabel 'Number of Reported Records'
set ylabel 'Time Per Reported Record (microsec)'
set xtics 1,10,100000
set ytics 1,2,1024
set output 'CellQuerySizeRandomSphereScaled.eps'
plot [1.26/1.1:376226.*1.1]\
  'CellQuerySizeRandomSphere.dat'\
  using 2:($3/$2*10e6) title "Cell Array" with linespoints,\
  'CellQuerySizeRandomSphere.dat'\
  using 2:($4/$2*10e6) title "Sparse Cell Array" with linespoints,\
  'CellQuerySizeRandomSphere.dat'\
  using 2:($5/$2*10e6) title "Cells w/ Binary Searching" with linespoints
# CellQuerySizeRandomCube.gp

set size 0.7,0.7
set key right bottom
set title 'Cell Methods, Random Points in a Cube'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set logscale xy
set xtics 0.01,2,0.64
set nomxtics
set ytics 0.00001,10,0.1
set nomytics
set output 'CellQuerySizeRandomCubeTotal.eps'
plot [0.00984313/1.1:0.793701*1.1]\
  'CellQuerySizeRandomCube.dat'\
  using 1:3 title "Cell Array" with linespoints,\
  'CellQuerySizeRandomCube.dat'\
  using 1:4 title "Sparse Cell Array" with linespoints,\
  'CellQuerySizeRandomCube.dat'\
  using 1:5 title "Cells w/ Binary Searching" with linespoints,\
  'CellQuerySizeRandomCube.dat'\
  using 1:6 title "Sequential Scan" with linespoints

set key right top
set xlabel 'Number of Reported Records'
set ylabel 'Time Per Reported Record (microsec)'
set xtics 1,10,100000
set ytics 1,2,1024
set output 'CellQuerySizeRandomCubeScaled.eps'
plot [0.953674/1.1:500000.*1.1]\
  'CellQuerySizeRandomCube.dat'\
  using 2:($3/$2*10e6) title "Cell Array" with linespoints,\
  'CellQuerySizeRandomCube.dat'\
  using 2:($4/$2*10e6) title "Sparse Cell Array" with linespoints,\
  'CellQuerySizeRandomCube.dat'\
  using 2:($5/$2*10e6) title "Cells w/ Binary Searching" with linespoints
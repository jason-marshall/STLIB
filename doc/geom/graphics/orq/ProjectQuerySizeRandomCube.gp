# ProjectQuerySizeRandomCube.gp

set size 0.5,0.5
set key right bottom
set title 'Projection Methods, Random in Cube'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set logscale xy
set xtics 0.01,4,0.64
set nomxtics
set ytics 0.005,2,0.32
set nomytics
set output 'ProjectQuerySizeRandomCubeTotal.eps'
plot [0.00984313/1.1:0.793701*1.1]\
  'ProjectQuerySizeRandomCube.dat'\
  using 1:3 title "Projection"with linespoints,\
  'ProjectQuerySizeRandomCube.dat'\
  using 1:4 title "Point-In-Box" with linespoints,\
  'ProjectQuerySizeRandomCube.dat'\
  using 1:5 title "Sequential Scan" with linespoints

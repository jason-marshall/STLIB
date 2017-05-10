# RandomCubeQuerySize.gp

set size 0.4,0.4
set nokey
set title 'Random Points in a Cube'
set xlabel 'Query Size'
set ylabel 'Reported Records'
set terminal postscript eps color
set output 'RandomCubeQuerySize.eps'
set logscale xy
set xtics 0.01,4,0.64
set nomxtics
set ytics 1,10,100000
set nomytics
plot [0.00984313/1.1:0.793701*1.1] [0.953674/1.5:500000*1.5]\
  'RandomCubeQuerySize.dat' with points

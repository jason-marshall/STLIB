# RandomSphereQuerySize.gp

set size 0.4,0.4
set nokey
set title 'Random Points on a Sphere'
set xlabel 'Query Size'
set ylabel 'Reported Records'
set terminal postscript eps color
set output 'RandomSphereQuerySize.eps'
set logscale xy
set xtics 0.01,10,1
set nomxtics
set ytics 1,10,100000
set nomytics
plot [0.00138107/1.2:2*1.2] [1.26/1.5:376226.*1.5]\
  'RandomSphereQuerySize.dat' with points

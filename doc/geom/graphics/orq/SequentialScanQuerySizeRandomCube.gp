# SequentialScanQuerySizeRandomCube.gp

set size 0.4,0.4
set nokey
set title 'Sequential Scan, Random in Cube'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'SequentialScanQuerySizeRandomCube.eps'
set logscale x
set xtics 0.01,4,0.64
set nomxtics
set ytics 0,0.05,0.30
set nomytics
plot [0.00984313/1.1:0.793701*1.1] [0:0.250511*1.04]\
  'SequentialScanQuerySizeRandomCube.dat' with linespoints

# SequentialScanQuerySizeRandomSphere.gp

set size 0.4,0.4
set nokey
set title 'Sequential Scan, Random on Sphere'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set output 'SequentialScanQuerySizeRandomSphere.eps'
set logscale x
set xtics 0.01,10,1
set nomxtics
set ytics 0,0.05,0.30
set nomytics
plot [0.00138107/1.1:2*1.1] [0:0.233221*1.04]\
  'SequentialScanQuerySizeRandomSphere.dat' with linespoints

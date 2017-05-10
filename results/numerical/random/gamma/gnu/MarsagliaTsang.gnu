# MarsagliaTsang.gnu

set title "Gamma deviates with the method from Marsaglia and Tsang."
set xlabel "Shape"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "MarsagliaTsang.jpg"
plot [1e0:1e2] [0:]\
'MarsagliaTsang.txt' title "Marsaglia/Tsang" with linespoints
set terminal postscript eps 22 color
set output "MarsagliaTsang.eps"
replot


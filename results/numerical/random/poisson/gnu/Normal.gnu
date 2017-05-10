# Normal.gnu

set title "Poisson deviates with the normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "Normal.jpg"
plot [1e2:1e6] [0:]\
'Normal.txt' title "normal approximation" with linespoints
set terminal postscript eps 22 color
set output "Normal.eps"
replot


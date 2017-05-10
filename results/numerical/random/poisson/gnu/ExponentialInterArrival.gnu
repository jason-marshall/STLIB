# ExponentialInterArrival.gnu

set title "Poisson deviates with the exponential inter-arrival method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "ExponentialInterArrival.jpg"
plot [1e-2:1e1] [0:]\
'ExponentialInterArrival.txt' title "Exponential" with linespoints
set terminal postscript eps 22 color
set output "ExponentialInterArrival.eps"
replot


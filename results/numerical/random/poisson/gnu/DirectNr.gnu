# DirectNr.gnu

set title "Poisson deviates with the direct method from Numerical Recipes."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "DirectNr.jpg"
plot [1e-2:1e1] [0:]\
'DirectNr.txt' title "Direct NR" with linespoints, \
'ExponentialInterArrivalUsingUniform.txt' title "Exp Int-Arr Using Uniform" with linespoints
set terminal postscript eps 22 color
set output "DirectNr.eps"
replot


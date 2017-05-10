# ExponentialInterArrivalUsingUniform.gnu

set title "Poisson deviates with the exponential inter-arrival method using uniform deviates."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "ExponentialInterArrivalUsingUniform.jpg"
plot [1e-2:1e1] [0:]\
'ExponentialInterArrivalUsingUniform.txt' title "Default" with linespoints, \
'ExponentialInterArrivalUsingUniformAppExp.txt' title "App Exp" with linespoints, \
'ExponentialInterArrivalUsingUniformCache.txt' title "Cache" with linespoints, \
'ExponentialInterArrivalUsingUniformSmall.txt' title "Small" with linespoints, \
'ExponentialInterArrivalUsingUniformCacheSmall.txt' title "Cache Small" with linespoints
set terminal postscript eps 22 color
set output "ExponentialInterArrivalUsingUniform.eps"
replot


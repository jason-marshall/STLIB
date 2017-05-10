# InversionFromModeChopDown.gnu

set title "Poisson deviates with the inversion from the mode (chop down) method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionFromModeChopDown.jpg"
plot [1e-2:1e2] [0:]\
'InversionFromModeChopDown.txt' title "Default" with linespoints, \
'InversionFromModeChopDownApprox.txt' title "Approximate" with linespoints, \
'InversionFromModeChopDownCache.txt' title "Cache" with linespoints, \
'InversionFromModeChopDownSmall.txt' title "Small" with linespoints, \
'InversionFromModeChopDownCacheSmall.txt' title "Cache Small" with linespoints
set terminal postscript eps 22 color
set output "InversionFromModeChopDown.eps"
replot


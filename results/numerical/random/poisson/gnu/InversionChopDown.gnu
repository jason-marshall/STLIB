# InversionChopDown.gnu

set title "Poisson deviates with the inversion (chop down) method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionChopDown.jpg"
plot [1e-2:2e1] [0:]\
'InversionChopDown.txt' title "Default" with linespoints, \
'InversionChopDownAppExp.txt' title "App Exp" with linespoints, \
'InversionChopDownCache.txt' title "Cache" with linespoints, \
'InversionChopDownSmall.txt' title "Small" with linespoints, \
'InversionChopDownZero.txt' title "Zero" with linespoints, \
'InversionChopDownCacheSmall.txt' title "Cache Small" with linespoints, \
'InversionChopDownSmallZero.txt' title "Small Zero" with linespoints
set terminal postscript eps 22 color
set output "InversionChopDown.eps"
replot


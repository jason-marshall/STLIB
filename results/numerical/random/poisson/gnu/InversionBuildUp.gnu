# InversionBuildUp.gnu

set title "Poisson deviates with the inversion (build up) method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionBuildUp.jpg"
plot [1e-2:2e1] [0:]\
'InversionBuildUp.txt' title "Default" with linespoints, \
'InversionBuildUpCache.txt' title "Cache" with linespoints, \
'InversionBuildUpSmall.txt' title "Small" with linespoints, \
'InversionBuildUpCacheSmall.txt' title "Cache Small" with linespoints
set terminal postscript eps 22 color
set output "InversionBuildUp.eps"
replot


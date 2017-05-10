# CompareSmallArgument.gnu

set title "Best methods for small arguments."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareSmallArgument.jpg"
plot [1e-2:1e1] [0:300]\
'ExponentialInterArrival.txt' title "Exponential" with linespoints, \
'InversionChopDown.txt' title "Inversion chop-down" with linespoints, \
'InversionChopDownAppExp.txt' title "Inversion chop-down (app exp)" with linespoints, \
'InversionChopDownCache.txt' title "Inversion chop-down (cache)" with linespoints, \
'InversionChopDownSmall.txt' title "Inversion chop-down (small)" with linespoints
set terminal postscript eps 22 color
set output "CompareSmallArgument.eps"
replot


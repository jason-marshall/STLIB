# StochKit.gnu

set title "Poisson deviates with the StochKit library."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "StochKitSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'StochKit.txt' title "StochKit" with linespoints
set terminal postscript eps 22 color
set output "StochKitSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "StochKitLargeArgument.jpg"
plot [1e1:1e7] [0:]\
'StochKit.txt' title "StochKit" with linespoints
set terminal postscript eps 22 color
set output "StochKitLargeArgument.eps"
replot


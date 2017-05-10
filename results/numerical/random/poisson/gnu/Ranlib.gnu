# Ranlib.gnu

set title "Poisson deviates with the Ranlib library."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "RanlibSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'Ranlib.txt' title "Ranlib" with linespoints
set terminal postscript eps 22 color
set output "RanlibSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "RanlibLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'Ranlib.txt' title "Ranlib" with linespoints
set terminal postscript eps 22 color
set output "RanlibLargeArgument.eps"
replot


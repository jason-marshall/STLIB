# InversionBuildUpSimple.gnu

set title "Poisson deviates with the inversion (build up) method.\nComparison of simple and efficient implementations."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionBuildUpSimple.jpg"
plot [1e-2:2e1] [0:]\
'InversionBuildUpSimple.txt' title "Simple" with linespoints, \
'InversionBuildUp.txt' title "Efficient" with linespoints
set terminal postscript eps 22 color
set output "InversionBuildUpSimple.eps"
replot


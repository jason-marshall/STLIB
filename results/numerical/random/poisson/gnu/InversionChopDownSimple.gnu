# InversionChopDownSimple.gnu

set title "Poisson deviates with the inversion (chop down) method.\nComparison of simple and efficient implementations."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionChopDownSimple.jpg"
plot [1e-2:2e1] [0:]\
'InversionChopDownSimple.txt' title "Simple" with linespoints, \
'InversionChopDown.txt' title "Efficient" with linespoints
set terminal postscript eps 22 color
set output "InversionChopDownSimple.eps"
replot


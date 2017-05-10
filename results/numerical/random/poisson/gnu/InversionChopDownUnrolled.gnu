# InversionChopDownUnrolled.gnu

set title "Poisson deviates with the unrolled inversion (chop down) method.\nComparison with standard efficient implementation."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionChopDownUnrolled.jpg"
plot [1e-2:2e1] [0:]\
'InversionChopDownUnrolled.txt' title "Unrolled" with linespoints, \
'InversionChopDown.txt' title "Standard" with linespoints
set terminal postscript eps 22 color
set output "InversionChopDownUnrolled.eps"
replot


# DirectRejectionNr.gnu

set title "Poisson deviates with the direct/rejection method from Numerical Recipes."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "DirectRejectionNrSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'DirectRejectionNr.txt' title "direct/rejection" with linespoints
set terminal postscript eps 22 color
set output "DirectRejectionNrSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "DirectRejectionNrLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'DirectRejectionNr.txt' title "direct/rejection" with linespoints
set terminal postscript eps 22 color
set output "DirectRejectionNrLargeArgument.eps"
replot


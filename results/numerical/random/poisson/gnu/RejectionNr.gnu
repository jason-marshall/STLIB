# RejectionNr.gnu

set title "Poisson deviates with the rejection method from Numerical Recipes."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "RejectionNr.jpg"
plot [1e0:1e6] [0:]\
'RejectionNr.txt' title "rejection" with linespoints
set terminal postscript eps 22 color
set output "RejectionNr.eps"
replot


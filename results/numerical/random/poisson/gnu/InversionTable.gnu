# InversionTable.gnu

set title "Poisson deviates with the table inversion method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionTable.jpg"
plot [1e-2:2e1] [0:]\
'InversionTable.txt' title "Table inversion" with linespoints
set terminal postscript eps 22 color
set output "InversionTable.eps"
replot


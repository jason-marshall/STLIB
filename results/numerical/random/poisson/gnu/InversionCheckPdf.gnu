# InversionCheckPdf.gnu

set title "Poisson deviates with the inversion (check PDF) method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionCheckPdf.jpg"
plot [1e-2:2e1] [0:]\
'InversionCheckPdf.txt' title "Inversion check PDF" with linespoints
set terminal postscript eps 22 color
set output "InversionCheckPdf.eps"
replot


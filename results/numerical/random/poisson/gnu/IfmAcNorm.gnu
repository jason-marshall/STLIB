# IfmAcNorm.gnu

set title "Poisson deviates with the inversion from the mode\n/acceptance-complement/normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "IfmAcNormSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'IfmAcNorm.txt' title "Default" with linespoints
set terminal postscript eps 22 color
set output "IfmAcNormSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "IfmAcNormLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'IfmAcNorm.txt' title "Default" with linespoints
set terminal postscript eps 22 color
set output "IfmAcNormLargeArgument.eps"
replot


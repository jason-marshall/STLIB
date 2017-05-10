# InvIfmAcNorm.gnu

set title "Poisson deviates with the inversion/inversion from the mode\n/acceptance-complement/normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InvIfmAcNormSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InvIfmAcNorm.txt' title "Default" with linespoints
set terminal postscript eps 22 color
set output "InvIfmAcNormSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "InvIfmAcNormLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InvIfmAcNorm.txt' title "Default" with linespoints
set terminal postscript eps 22 color
set output "InvIfmAcNormLargeArgument.eps"
replot


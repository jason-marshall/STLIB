# InvAcNorm.gnu

set title "Poisson deviates with the inversion/acceptance-complement/normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InvAcNormSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InvAcNorm.txt' title "Default" with linespoints, \
'InvAcNormSmall.txt' title "Small" with linespoints
set terminal postscript eps 22 color
set output "InvAcNormSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "InvAcNormLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InvAcNorm.txt' title "Default" with linespoints, \
'InvAcNormSmall.txt' title "Small" with linespoints
set terminal postscript eps 22 color
set output "InvAcNormLargeArgument.eps"
replot


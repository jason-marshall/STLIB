# ExpAcNorm.gnu

set title "Poisson deviates with the exponential/acceptance-complement/normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "ExpAcNormSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'ExpAcNorm.txt' title "Exp/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "ExpAcNormSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "ExpAcNormLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'ExpAcNorm.txt' title "Exp/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "ExpAcNormLargeArgument.eps"
replot


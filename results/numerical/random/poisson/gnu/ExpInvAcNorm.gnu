# ExpInvAcNorm.gnu

set title "Poisson deviates with the exponential/inversion/\nacceptance-complement/normal approximation method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "ExpInvAcNormSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "ExpInvAcNormSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "ExpInvAcNormLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "ExpInvAcNormLargeArgument.eps"
replot


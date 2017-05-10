# ExpInvAc.gnu

set title "Poisson deviates with the exponential/inversion/\nacceptance-complement method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "ExpInvAcSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'ExpInvAc.txt' title "Exp/Inv/Acc-Comp" with linespoints
set terminal postscript eps 22 color
set output "ExpInvAcSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "ExpInvAcLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'ExpInvAc.txt' title "Exp/Inv/Acc-Comp" with linespoints
set terminal postscript eps 22 color
set output "ExpInvAcLargeArgument.eps"
replot


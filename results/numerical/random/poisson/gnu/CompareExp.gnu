# CompareExp.gnu

set title "Best overall methods that start with exponential inter-arrival."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareExpSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'ExpAcNorm.txt' title "Exp/Acc-Comp/Norm" with linespoints, \
'ExpInvAc.txt' title "Exp/Inv/Acc-Comp" with linespoints, \
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareExpSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "CompareExpLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'ExpAcNorm.txt' title "Exp/Acc-Comp/Norm" with linespoints, \
'ExpInvAc.txt' title "Exp/Inv/Acc-Comp" with linespoints, \
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareExpLargeArgument.eps"
replot


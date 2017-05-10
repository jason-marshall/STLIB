# CompareInv.gnu

set title "Best overall methods that start with inversion."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareInvSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InvAcNorm.txt' title "Inv/Acc-Comp/Norm" with linespoints, \
'InvAcNormSmall.txt' title "Inv/Acc-Comp/Norm (small)" with linespoints, \
'InvIfmAcNorm.txt' title "Inv/IFM/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareInvSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "CompareInvLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InvAcNorm.txt' title "Inv/Acc-Comp/Norm" with linespoints, \
'InvAcNormSmall.txt' title "Inv/Acc-Comp/Norm (small)" with linespoints, \
'InvIfmAcNorm.txt' title "Inv/IFM/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareInvLargeArgument.eps"
replot


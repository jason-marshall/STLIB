# CompareOverall.gnu

set title "Best overall methods."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareOverallSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints, \
'InvIfmAcNorm.txt' title "Inv/IFM/Acc-Comp/Norm" with linespoints, \
'IfmAcNorm.txt' title "IFM/Acc-Comp/Norm" with linespoints, \
'InversionTableAcceptanceComplementWinrand.txt' title "Inv Table/Acc-Comp" with linespoints
set terminal postscript eps 22 color
set output "CompareOverallSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "CompareOverallLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'ExpInvAcNorm.txt' title "Exp/Inv/Acc-Comp/Norm" with linespoints, \
'InvIfmAcNorm.txt' title "Inv/IFM/Acc-Comp/Norm" with linespoints, \
'IfmAcNorm.txt' title "IFM/Acc-Comp/Norm" with linespoints, \
'InversionTableAcceptanceComplementWinrand.txt' title "Inv Table/Acc-Comp" with linespoints
set terminal postscript eps 22 color
set output "CompareOverallLargeArgument.eps"
replot


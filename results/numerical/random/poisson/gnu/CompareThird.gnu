# CompareThird.gnu

set title "Best overall third-party methods."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareThirdSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'Ranlib.txt' title "Ranlib" with linespoints, \
'InversionTableAcceptanceComplementWinrand.txt' title "Inv Table/Acc-Comp" with linespoints, \
'StochKit.txt' title "StochKit" with linespoints
set terminal postscript eps 22 color
set output "CompareThirdSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "CompareThirdLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'Ranlib.txt' title "Ranlib" with linespoints, \
'InversionTableAcceptanceComplementWinrand.txt' title "Inv Table/Acc-Comp" with linespoints, \
'StochKit.txt' title "StochKit" with linespoints
set terminal postscript eps 22 color
set output "CompareThirdLargeArgument.eps"
replot


# CompareIfm.gnu

set title "Best overall methods that start with inversion from the mode."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareIfmSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'IfmAcNorm.txt' title "IFM/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareIfmSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "CompareIfmLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'IfmAcNorm.txt' title "IFM/Acc-Comp/Norm" with linespoints
set terminal postscript eps 22 color
set output "CompareIfmLargeArgument.eps"
replot


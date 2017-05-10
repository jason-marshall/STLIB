# CompareLargeArgument.gnu

set title "Best methods for large arguments."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "CompareLargeArgument.jpg"
plot [1e0:1e6] [0:]\
'AcceptanceComplementWinrand.txt' title "acceptance-complement" with linespoints, \
'Ranlib.txt' title "Ranlib" with linespoints
set terminal postscript eps 22 color
set output "CompareLargeArgument.eps"
replot


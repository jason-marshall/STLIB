# InversionTableAcceptanceComplementWinrand.gnu

set title "Poisson deviates with the table inversion/acceptance complement\nmethod from WinRand."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionTableAcceptanceComplementWinrandSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InversionTableAcceptanceComplementWinrand.txt' title "table inversion/acceptance complement" with linespoints
set terminal postscript eps 22 color
set output "InversionTableAcceptanceComplementWinrandSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "InversionTableAcceptanceComplementWinrandLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InversionTableAcceptanceComplementWinrand.txt' title "table inversion/acceptance complement" with linespoints
set terminal postscript eps 22 color
set output "InversionTableAcceptanceComplementWinrandLargeArgument.eps"
replot


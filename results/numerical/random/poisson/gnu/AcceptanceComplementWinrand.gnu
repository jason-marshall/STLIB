# AcceptanceComplementWinrand.gnu

set title "Poisson deviates with the acceptance-complement method from Winrand."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "AcceptanceComplementWinrand.jpg"
plot [1e0:1e6] [0:]\
'AcceptanceComplementWinrand.txt' title "acceptance-complement" with linespoints
set terminal postscript eps 22 color
set output "AcceptanceComplementWinrand.eps"
replot


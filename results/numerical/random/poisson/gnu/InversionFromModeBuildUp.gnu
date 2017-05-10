# InversionFromModeBuildUp.gnu

set title "Poisson deviates with the inversion from the mode (build up) method."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionFromModeBuildUp.jpg"
plot [1e-2:1e2] [0:]\
'InversionFromModeBuildUp.txt' title "Default" with linespoints, \
'InversionFromModeBuildUpApprox.txt' title "Approximate" with linespoints
set terminal postscript eps 22 color
set output "InversionFromModeBuildUp.eps"
replot


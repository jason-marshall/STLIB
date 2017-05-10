# InversionRejectionPatchworkWinrand.gnu

set title "Poisson deviates with the inversion/patchwork rejection\nmethod from WinRand."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionRejectionPatchworkWinrandSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InversionRejectionPatchworkWinrand.txt' title "inversion/patchwork rejection" with linespoints
set terminal postscript eps 22 color
set output "InversionRejectionPatchworkWinrandSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "InversionRejectionPatchworkWinrandLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InversionRejectionPatchworkWinrand.txt' title "inversion/patchwork rejection" with linespoints
set terminal postscript eps 22 color
set output "InversionRejectionPatchworkWinrandLargeArgument.eps"
replot


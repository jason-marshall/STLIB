# InversionRatioOfUniformsWinrand.gnu

set title "Poisson deviates with the inversion/ratio of uniforms\nmethod from WinRand."
set xlabel "Mean"
set ylabel "Execution time (in nanoseconds)"
set key bottom

set logscale x
set terminal jpeg
set output "InversionRatioOfUniformsWinrandSmallArgument.jpg"
plot [1e-2:1e1] [0:]\
'InversionRatioOfUniformsWinrand.txt' title "inversion/ratio of uniforms" with linespoints
set terminal postscript eps 22 color
set output "InversionRatioOfUniformsWinrandSmallArgument.eps"
replot

set logscale x
set terminal jpeg
set output "InversionRatioOfUniformsWinrandLargeArgument.jpg"
plot [1e1:1e6] [0:]\
'InversionRatioOfUniformsWinrand.txt' title "inversion/ratio of uniforms" with linespoints
set terminal postscript eps 22 color
set output "InversionRatioOfUniformsWinrandLargeArgument.eps"
replot


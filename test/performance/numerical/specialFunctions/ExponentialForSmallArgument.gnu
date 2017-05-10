# First run ExponentialForSmallArgument.exe to get the data files.

set logscale x
set title "Execution times (in nanoseconds) for computing the exponential."
set key bottom
set terminal jpeg
set output "ExponentialForSmallArgument.jpg"
plot [1e-16:10] [0:90] 'ExponentialForSmallArgument.txt' \
title "ExponentialForSmallArgument" with lines, \
'stdexp.txt' title "std::exp()" with lines
set terminal postscript eps 22 color
set output "ExponentialForSmallArgument.eps"
replot

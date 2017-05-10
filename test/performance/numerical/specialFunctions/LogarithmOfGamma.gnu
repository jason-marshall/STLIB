# First run LogarithmOfGamma.exe to get the data file.

set logscale x
set title "Execution times (in nanoseconds) for log(Gamma(x))."
set key bottom
set terminal jpeg
set output "LogarithmOfGamma.jpg"
plot [1e-16:1e8] 'LogarithmOfGamma.txt' notitle with lines
set terminal postscript eps 22 color
set output "LogarithmOfGamma.eps"
replot

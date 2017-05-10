set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "top_bad.eps"
plot 'top_bad.dat' with lines

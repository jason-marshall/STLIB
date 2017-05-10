set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "size_small.eps"
plot 'size_small.dat' with lines

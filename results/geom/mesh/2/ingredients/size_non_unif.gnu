set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "size_non_unif.eps"
plot 'size_non_unif.dat' with lines

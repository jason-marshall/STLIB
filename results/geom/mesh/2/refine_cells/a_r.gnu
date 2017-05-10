set size ratio 1
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "a_r.eps"
plot 'a_r.dat' with lines

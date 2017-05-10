set size ratio 0.1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "wedge_r.eps"
plot 'wedge_r.dat' with lines

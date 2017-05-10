set size ratio 5/6 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "nautilus.eps"
plot 'nautilus.dat' with lines

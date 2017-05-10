set size ratio 1
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "flipped.eps"
plot 'flipped.dat' with lines

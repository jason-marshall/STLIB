set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "geom_bad.eps"
plot 'geom_bad.dat' with lines

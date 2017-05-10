set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "boundary_initial.eps"
plot 'boundary_initial.dat' with lines, 'boundary_boundary.dat' with lines

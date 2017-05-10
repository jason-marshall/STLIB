set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "boundary_moved.eps"
plot 'boundary_moved.dat' with lines, 'boundary_boundary.dat' with lines

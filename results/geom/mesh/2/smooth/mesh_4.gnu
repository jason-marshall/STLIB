set size ratio (sqrt(3) / 2) 0.75, 0.75
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_4.eps"
plot 'mesh_4.dat' with lines linewidth 2

set size ratio 1
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_0.1.eps"
plot 'mesh_0.1.dat' with lines
